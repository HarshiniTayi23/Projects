import { useState, useEffect, useRef } from 'react'
import { useQuery, useMutation, useSubscription } from '@apollo/client'
import { gql } from '@apollo/client'
import { useUserData } from '@nhost/react'
import { Send, Bot, User, Loader2 } from 'lucide-react'

const GET_CHAT = gql`
  query GetChat($chatId: uuid!) {
    chats_by_pk(id: $chatId) {
      id
      title
      created_at
    }
  }
`

const GET_MESSAGES = gql`
  query GetMessages($chatId: uuid!) {
    messages(
      where: { chat_id: { _eq: $chatId } }
      order_by: { created_at: asc }
    ) {
      id
      content
      is_from_user
      created_at
      user_id
    }
  }
`

const MESSAGES_SUBSCRIPTION = gql`
  subscription MessagesSubscription($chatId: uuid!) {
    messages(
      where: { chat_id: { _eq: $chatId } }
      order_by: { created_at: asc }
    ) {
      id
      content
      is_from_user
      created_at
      user_id
    }
  }
`

const ADD_MESSAGE = gql`
  mutation AddMessage($chatId: uuid!, $content: String!, $isFromUser: Boolean!) {
    insert_messages_one(object: {
      chat_id: $chatId
      content: $content
      is_from_user: $isFromUser
    }) {
      id
      content
      is_from_user
      created_at
    }
  }
`

const SEND_MESSAGE_ACTION = gql`
  mutation SendMessageAction($chatId: uuid!, $message: String!) {
    sendMessage(chatId: $chatId, message: $message) {
      success
      message
    }
  }
`

export default function ChatView({ chatId }) {
  const [newMessage, setNewMessage] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const messagesEndRef = useRef(null)
  const user = useUserData()

  const { data: chatData } = useQuery(GET_CHAT, {
    variables: { chatId }
  })

  const { data: messagesData, loading: messagesLoading } = useSubscription(MESSAGES_SUBSCRIPTION, {
    variables: { chatId }
  })

  const [addMessage] = useMutation(ADD_MESSAGE)
  const [sendMessageAction, { loading: actionLoading }] = useMutation(SEND_MESSAGE_ACTION)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messagesData?.messages])

  const handleSendMessage = async (e) => {
    e.preventDefault()
    if (!newMessage.trim() || actionLoading) return

    const messageContent = newMessage.trim()
    setNewMessage('')
    setIsTyping(true)

    try {
      // Add user message to database
      await addMessage({
        variables: {
          chatId,
          content: messageContent,
          isFromUser: true
        }
      })

      // Call Hasura Action to trigger chatbot
      await sendMessageAction({
        variables: {
          chatId,
          message: messageContent
        }
      })
    } catch (error) {
      console.error('Error sending message:', error)
      // You might want to show an error toast here
    } finally {
      setIsTyping(false)
    }
  }

  const formatTime = (dateString) => {
    return new Date(dateString).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    })
  }

  if (messagesLoading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin text-indigo-600 mx-auto mb-2" />
          <p className="text-gray-600">Loading messages...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 flex flex-col h-full">
      {/* Chat Header */}
      <div className="bg-white border-b border-gray-200 p-4">
        <h2 className="text-lg font-semibold text-gray-900" data-testid={`chat-header-${chatId}`}>
          {chatData?.chats_by_pk?.title || 'Chat'}
        </h2>
        <p className="text-sm text-gray-500">
          Created {chatData?.chats_by_pk?.created_at && new Date(chatData.chats_by_pk.created_at).toLocaleDateString()}
        </p>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
        {messagesData?.messages?.length === 0 ? (
          <div className="text-center text-gray-500 mt-8">
            <Bot className="h-12 w-12 text-gray-300 mx-auto mb-2" />
            <p>Start a conversation with the chatbot!</p>
          </div>
        ) : (
          messagesData?.messages?.map((message) => (
            <div
              key={message.id}
              data-testid={`message-${message.id}`}
              className={`flex ${message.is_from_user ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                  message.is_from_user
                    ? 'bg-indigo-600 text-white'
                    : 'bg-white text-gray-900 border border-gray-200'
                }`}
              >
                <div className="flex items-start space-x-2">
                  <div className={`flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center ${
                    message.is_from_user ? 'bg-indigo-500' : 'bg-gray-100'
                  }`}>
                    {message.is_from_user ? (
                      <User className="h-3 w-3 text-white" />
                    ) : (
                      <Bot className="h-3 w-3 text-gray-600" />
                    )}
                  </div>
                  <div className="flex-1">
                    <p className="text-sm" data-testid={`message-content-${message.id}`}>
                      {message.content}
                    </p>
                    <p className={`text-xs mt-1 ${
                      message.is_from_user ? 'text-indigo-200' : 'text-gray-500'
                    }`}>
                      {formatTime(message.created_at)}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          ))
        )}

        {/* Typing indicator */}
        {isTyping && (
          <div className="flex justify-start">
            <div className="max-w-xs lg:max-w-md px-4 py-2 rounded-lg bg-white text-gray-900 border border-gray-200">
              <div className="flex items-center space-x-2">
                <Bot className="h-4 w-4 text-gray-600" />
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                </div>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Message Input */}
      <div className="bg-white border-t border-gray-200 p-4">
        <form onSubmit={handleSendMessage} className="flex space-x-4">
          <div className="flex-1">
            <input
              type="text"
              value={newMessage}
              onChange={(e) => setNewMessage(e.target.value)}
              placeholder="Type your message..."
              disabled={actionLoading}
              data-testid="input-message"
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent disabled:opacity-50"
            />
          </div>
          <button
            type="submit"
            disabled={!newMessage.trim() || actionLoading}
            data-testid="button-send-message"
            className="px-6 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
          >
            {actionLoading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
            <span>{actionLoading ? 'Sending...' : 'Send'}</span>
          </button>
        </form>
      </div>
    </div>
  )
}