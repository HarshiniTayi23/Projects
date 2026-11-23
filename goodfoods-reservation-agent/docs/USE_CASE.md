# Use Case: Conversational Reservation Agent for GoodFoods

| | |
|---|---|
| **Client** | GoodFoods, a multi-cuisine restaurant chain with 75 outlets across Bengaluru |
| **Use case** | AI agent that discovers, recommends, books, modifies and cancels tables through natural conversation |
| **Channels** | Web chat (built), WhatsApp and phone voice (roadmap) |
| **Primary users** | Diners; outlet hosts and managers |
| **Owner** | GoodFoods Head of Guest Experience, with Operations and Marketing |

---

## 1. Problem statement

GoodFoods takes most reservations by **phone, walk-in and third-party aggregators**. As the chain grew, this created problems that get worse with every new outlet:

1. **Peak-hour phone overload.** Calls cluster from 6 to 8 pm, exactly when hosts are busiest seating guests. Calls go unanswered and those guests book a competitor.
2. **Each outlet sells only its own tables.** When one outlet is full, the host says "sorry". Nobody suggests the sister outlet 2 km away with free tables, so the chain loses demand it could have kept.
3. **No-shows and late cancellations** leave seats empty at peak (typically 10–15% of reservations in Indian casual dining).
4. **Aggregator dependence.** Platforms charge per cover or a commission, own the guest relationship, and show competitors on the same screen.
5. **No guest data.** Preferences, occasions and visit history sit in notebooks or in a platform's database, not with GoodFoods.
6. **Inconsistent experience.** Wording, upselling and accuracy depend on which host picks up.

## 2. Proposed solution

A conversational agent (persona "Gia") that runs on a small, low-cost open model and acts through a defined set of tools:

| Capability | What the guest experiences |
|---|---|
| Discovery | "Italian in Indiranagar with outdoor seating?" gives filtered options |
| Recommendation | "Somewhere quiet for our anniversary" gives the top 3 with a reason for each |
| Live availability | Real seat inventory per 90-minute slot |
| Booking | Collects only the missing details, reads back a summary, confirms, issues a code |
| Recovery when full | Offers nearby times **and sister outlets** with free tables |
| Self-service changes | Modify or cancel with the code and phone number for verification |

**How it's built (summary):** the LLM decides intent and which tool to call. Every business rule (capacity, opening hours, booking window, validation) is enforced in deterministic code behind an MCP server, so the model cannot overbook or invent availability. See the README for the technical design.

## 3. Business opportunities beyond "taking bookings"

These are the non-obvious sources of value, ranked by expected impact:

1. **Cross-outlet demand routing (keep the guest in the chain).** A standalone restaurant can't do this; a chain can. Every "we're full" becomes "Koramangala is full at 8, but our Indiranagar outlet has a table, 10 minutes away." This is already built in the alternatives logic.
2. **Off-peak demand shaping.** The agent can suggest 7:00 instead of 8:00, or Tuesday instead of Saturday, optionally with an incentive. This lifts table utilisation without adding seats.
3. **No-show reduction.** WhatsApp reminders with one-tap confirm or cancel, deposits for large parties at peak, and an automatic waitlist that backfills cancellations.
4. **First-party guest CRM.** Each conversation captures occasions (birthdays, anniversaries), dietary needs and preferred outlets. That supports targeted campaigns ("your anniversary is next week") without paying an aggregator.
5. **Occasion upsell at booking time.** "It's a birthday, would you like a cake (₹650) or a decorated table?" Pre-orders for large groups. This revenue is captured before the guest arrives.
6. **Large-party and event leads.** Groups above online limits are routed to private dining and the events team as qualified leads, not lost calls.
7. **Operational forecasting.** Booking curves by outlet, day and hour feed staffing rosters and ingredient purchasing, reducing labour cost and food waste.
8. **Language access.** Bengaluru diners speak Kannada, Hindi, Tamil, Telugu and English. A multilingual voice agent serves guests who would never use an app form.

## 4. Users and stakeholders

| Stakeholder | Interest | Role in the project |
|---|---|---|
| Diners (families, couples, office groups, tourists) | Fast, accurate booking at any hour in their language | End users; pilot feedback |
| Outlet hosts / floor managers | Fewer calls at peak; accurate covers forecast | Daily users of the booking view; escalation point |
| Head of Guest Experience | NPS, consistency | **Business owner** |
| Operations / COO | Table utilisation, staffing efficiency | Sponsor; defines capacity rules |
| Marketing / CRM | Guest data, campaigns, lower aggregator spend | Consumer of data; owns upsell offers |
| Finance / CFO | ROI, cost per booking | Approves budget; tracks benefits |
| IT / Tech | Integration with POS and table management, security | Integration owner |
| Legal / Compliance | DPDP Act 2023 (consent, retention), payment rules | Signs off on data handling |

## 5. Success metrics

Baselines are assumptions to be measured in the first 2 weeks of the pilot (shadow mode).

| Category | KPI | Baseline (assumed) | 6-month target |
|---|---|---|---|
| Adoption | Share of reservations made through the agent | 0% | 40% |
| Conversion | Conversations that end in a confirmed booking (when booking intent exists) | n/a | ≥ 65% |
| Recovery | "Full" situations converted to another time or outlet | ~0% | ≥ 35% |
| Accuracy | Bookings needing manual correction | n/a | < 1% |
| Accuracy | Correct tool choice on the intent eval set | n/a | ≥ 90% |
| Efficiency | Host phone minutes per outlet per day | ~300 min | −50% |
| Utilisation | Off-peak (before 7:30 pm) weekday seat utilisation | ~45% | 55% |
| Revenue protection | No-show rate | 12% | 7% |
| Experience | Post-booking CSAT | n/a | ≥ 4.5 / 5 |
| Experience | Median time to complete a booking | ~4 min by phone | < 90 s |
| Cost | AI cost per completed booking | n/a | < ₹2 |
| Strategic | Bookings via aggregators | ~30% | 20% |

## 6. ROI estimate (illustrative, chain of 75 outlets)

**Assumptions:** 60 reservations per outlet per day, 3 guests per booking, ₹700 average spend per guest, 55% contribution margin on extra covers (fixed costs already paid), host cost ₹250 per hour.

| Benefit lever | Logic | Annual value |
|---|---|---|
| Staff time | 5.25 h of call handling per outlet per day, 60% automated: 3.15 h × ₹250 × 75 × 365 | ₹2.2 Cr in cost avoided (or hours moved to guest service) |
| Recovered missed demand | 4 lost bookings per outlet per day, recover 50% = 2 × 3 × ₹700 × 75 × 365 = ₹11.5 Cr revenue | ₹6.3 Cr contribution |
| No-show reduction | 5-point drop = 3 bookings per day, half re-filled: 1.5 × 3 × ₹700 × 75 × 365 = ₹8.6 Cr revenue | ₹4.7 Cr contribution |
| Aggregator fees | 10% of bookings moved off aggregators: 6 × 3 guests × ₹40 per cover × 75 × 365 | ₹2.0 Cr saved |
| Upsell | 5% of bookings buy an add-on averaging ₹600: 3 × ₹600 × 75 × 365 × 50% margin | ₹2.5 Cr contribution |
| **Total** | | **≈ ₹17.7 Cr per year** |

**Costs:** build and integration ≈ ₹50 L (year 1); running costs ≈ ₹20 L per year (LLM inference ≈ ₹0.1 per conversation on an 8B model, hosting, WhatsApp/voice fees, a 1-person ops/QA role).

**Payback period: under 2 months after full rollout.** Even if only the two most conservative levers (staff time and aggregator fees, ≈ ₹4.1 Cr) materialise, ROI is above 5× in year 1.

## 7. Scope

**In scope (MVP, built):** discovery, recommendation, availability, booking, lookup, modify, cancel; 75 outlets; web chat; English.

**Phase 2:** WhatsApp channel, reminders, waitlist, deposits for large parties, host dashboard.

**Phase 3:** voice agent in Kannada, Hindi, Tamil and Telugu; CRM personalisation; integration with the POS and table-management system.

**Out of scope:** payments for the meal itself, delivery orders, loyalty-points engine.

## 8. Implementation timeline

| Weeks | Phase | Exit criteria |
|---|---|---|
| 0–2 | Discovery: map call flows, capacity rules per outlet, measure baselines | Signed-off rules and KPI baselines |
| 3–6 | Build and integrate: connect tools to the real table-management system, set up the WhatsApp Business API | Intent eval ≥ 90%; end-to-end tests green |
| 7–10 | **Pilot at 3 outlets** (different cuisines and sizes); hosts watch every booking | < 1% booking errors; CSAT ≥ 4.3 |
| 11–14 | Expand to 15 outlets; add reminders and waitlist | No-show rate falling; recovery ≥ 25% |
| 15–24 | Full rollout to 75 outlets; voice pilot in Kannada and Hindi | 40% of bookings through the agent |

## 9. Vertical expansion

The core abstraction is **"a resource with capacity × a time slot × a set of rules"**. The agent, MCP tool layer, evaluation harness and UI are reusable. Only the tool implementations and the prompt persona change.

| Vertical | Resource / slot | What changes | Example extra tool |
|---|---|---|---|
| **Other restaurant chains** (white-label) | Tables | Branding, outlet data | `pre_order_menu` |
| Cloud kitchens / QSR | Pickup slots | Kitchen throughput as capacity | `schedule_pickup` |
| Hotels | Rooms, spa, F&B outlets | Multi-night stays, rate plans | `quote_rate` |
| Salons and spas | Stylist × chair × service duration | Staff skills matrix | `book_with_stylist` |
| Clinics and diagnostics | Doctor slots, test slots | Consent, fasting instructions, stricter privacy | `send_prep_instructions` |
| Fitness studios | Class capacity | Memberships, waitlists | `join_waitlist` |
| Coworking / meeting rooms | Rooms × hours | Corporate billing | `invoice_company` |
| Banquet halls / events | Venue × date | Quotes, deposits | `generate_quote` |
| Auto service centres | Service bays | Vehicle details | `estimate_service_time` |

**Go-to-market path:** (1) GoodFoods as the reference customer, (2) white-label for mid-size Indian restaurant chains (10–200 outlets) that are too small to build this and unhappy with aggregator fees, (3) adjacent appointment-based businesses using the same platform.

**Potential customers:** mid-size casual-dining and QSR chains, hotel groups with several F&B outlets, salon chains, diagnostic-lab chains, and city tourism or heritage-site booking services.

## 10. Competitive advantages

1. **Chain-aware routing that keeps the guest in the chain.** Aggregators send overflow demand to competitors. This agent sends it to a sister outlet. A single-restaurant tool can't do this, and an aggregator has no reason to.
2. **Built for India's languages and channels, at small-model cost.** An 8B model with tool calling costs a fraction of a cent per conversation. It can run on sovereign or on-prem infrastructure and extends naturally to Indic-language voice and WhatsApp, where most Indian diners already are.
3. **Trustworthy by design and vendor-neutral.** Business rules are deterministic code, not prompts, so the model can't overbook or invent a confirmation (codes are checked against real tool results). Tools sit behind MCP, an open protocol, so the same agent can connect to any POS or table-management system, or be swapped for a better model, without a rewrite.

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Small model picks the wrong tool or invents details | Rules enforced in tools; confirm-before-write; codes checked against tool results; intent eval in CI |
| Double booking across channels (phone + agent) | Single source of truth for inventory; atomic check-and-insert transactions |
| Guest privacy (DPDP Act) | Collect only name and phone; mask phone in responses; retention policy; consent notice |
| Hosts resist the change | Hosts get a dashboard and override rights; pilot with champions; measure time saved for them |
| LLM provider outage | Retries with backoff; provider-agnostic client (any OpenAI-compatible API); graceful fallback message with the outlet phone |
| Prompt injection / abuse | Scope limited to reservations; tools are the only actions; input length limits; verification for changes |

## 12. Assumptions

- Seat capacity is modelled per outlet (total seats per 90-minute slot), not per physical table.
- All outlets are in Bengaluru, in one timezone (IST).
- Financial figures are planning estimates, to be replaced by pilot measurements.
- Guests identify themselves by phone number; no login.

## 13. Future enhancements

- Table-level seating (2-tops, 4-tops, combinable tables) and dynamic slot lengths by party size.
- WhatsApp and voice channels; Indic language support.
- Reminders, waitlist, deposits, cake and decoration add-ons.
- Guest profiles and personalised recommendations.
- Manager dashboard: live covers, no-show risk, demand forecast.
- A/B testing of prompts and recommendation weights against conversion.
