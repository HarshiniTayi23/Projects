"""Agent loop tests with a scripted (fake) LLM — deterministic, no network."""
import json

from agent.agent import Agent, FALLBACK_LLM_DOWN, FALLBACK_LOOP
from agent.config import Settings
from agent.llm_client import LLMError, LLMResponse
from tests.fakes import ScriptedLLM, ToyTools, tool_call

CFG = Settings(api_key="x", max_tool_iterations=4, history_turns=3)


def make(*responses):
    llm, tools = ScriptedLLM(*responses), ToyTools()
    return Agent(llm, tools, "You are a test agent.", CFG), llm, tools


def test_model_chooses_tool_and_result_is_fed_back():
    agent, llm, tools = make(
        LLMResponse(None, [tool_call("get_time", {"city": "Mumbai"})]),
        LLMResponse("It's 19:30 in Mumbai."),
    )
    turn = agent.chat("What time is it in Mumbai?")
    assert turn.reply == "It's 19:30 in Mumbai."
    assert tools.invocations == [("get_time", {"city": "Mumbai"})]
    second_request = llm.calls[1]
    assert second_request[-1]["role"] == "tool" and "19:30" in second_request[-1]["content"]


def test_no_tool_when_model_answers_directly():
    agent, _, tools = make(LLMResponse("Why did the chef quit? He lost his thyme."))
    turn = agent.chat("Tell me a joke")
    assert tools.invocations == [] and turn.tool_events == []


def test_invalid_json_arguments_are_reported_back_to_model():
    agent, llm, tools = make(
        LLMResponse(None, [tool_call("get_time", "{city: Mumbai")]),
        LLMResponse(None, [tool_call("get_time", {"city": "Mumbai"})]),
        LLMResponse("19:30"),
    )
    turn = agent.chat("time in Mumbai")
    assert turn.reply == "19:30"
    assert turn.tool_events[0].ok is False
    assert "INVALID_ARGUMENTS" in llm.calls[1][-1]["content"]


def test_unknown_tool_is_rejected_without_crashing():
    agent, _, tools = make(LLMResponse(None, [tool_call("delete_db", {})]), LLMResponse("ok"))
    turn = agent.chat("hi")
    assert tools.invocations == [] and "UNKNOWN_TOOL" in turn.tool_events[0].result


def test_infinite_tool_loop_stops_with_fallback():
    loop = [LLMResponse(None, [tool_call("get_time", {"city": "X"})]) for _ in range(CFG.max_tool_iterations)]
    agent, _, _ = make(*loop)
    turn = agent.chat("loop forever")
    assert turn.reply == FALLBACK_LOOP and turn.error


def test_llm_outage_gives_friendly_message():
    agent, _, _ = make(LLMError("503"))
    turn = agent.chat("hello")
    assert turn.reply == FALLBACK_LLM_DOWN


def test_parallel_tool_calls_are_all_executed():
    agent, _, tools = make(
        LLMResponse(None, [tool_call("get_time", {"city": "A"}), tool_call("get_time", {"city": "B"})]),
        LLMResponse("done"),
    )
    agent.chat("time in A and B")
    assert [a["city"] for _, a in tools.invocations] == ["A", "B"]


def test_history_is_trimmed_at_user_boundaries():
    agent, llm, _ = make(*[LLMResponse(f"r{i}") for i in range(5)])
    for i in range(5):
        agent.chat(f"msg {i}")
    sent = llm.calls[-1]
    assert sent[0]["role"] == "system"
    assert sent[1]["role"] == "user" and sent[1]["content"] == "msg 2"


def test_empty_input_does_not_call_model():
    agent, llm, _ = make()
    assert "help" in agent.chat("   ").reply and llm.calls == []


def test_hallucinated_confirmation_code_is_blocked():
    from agent.agent import UNGROUNDED_CODE_REPLY
    agent, _, _ = make(LLMResponse("All set! Your confirmation code is GF-ABCDE."))
    assert agent.chat("book it").reply == UNGROUNDED_CODE_REPLY


def test_real_confirmation_code_is_allowed():
    class BookingTools(ToyTools):
        def call(self, name, arguments):
            return json.dumps({"ok": True, "confirmation_code": "GF-K7P2Q"})
    llm = ScriptedLLM(LLMResponse(None, [tool_call("get_time", {"city": "x"})]), LLMResponse("Booked: GF-K7P2Q"))
    agent = Agent(llm, BookingTools(), "sys", CFG)
    assert agent.chat("book").reply == "Booked: GF-K7P2Q"


def test_code_typed_by_user_is_not_flagged():
    agent, _, _ = make(LLMResponse("Let me look up GF-7K2PQ for you. What's the phone number?"))
    assert "GF-7K2PQ" in agent.chat("status of GF-7K2PQ?").reply


def test_huge_tool_result_is_truncated():
    class BigTools(ToyTools):
        def call(self, name, arguments):
            return json.dumps({"ok": True, "blob": "x" * 20000})
    llm = ScriptedLLM(LLMResponse(None, [tool_call("get_time", {"city": "x"})]), LLMResponse("ok"))
    agent = Agent(llm, BigTools(), "sys", CFG)
    agent.chat("go")
    assert len(llm.calls[1][-1]["content"]) < 4100


def test_tool_exception_is_contained():
    class Boom(ToyTools):
        def call(self, name, arguments):
            raise RuntimeError("db locked")
    llm = ScriptedLLM(LLMResponse(None, [tool_call("get_time", {"city": "x"})]), LLMResponse("Sorry, try again."))
    turn = Agent(llm, Boom(), "sys", CFG).chat("go")
    assert turn.reply == "Sorry, try again." and "TOOL_FAILURE" in turn.tool_events[0].result
