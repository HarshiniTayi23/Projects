import { useState } from 'react'
import { useQuery, useMutation } from '@apollo/client'
import { gql } from '@apollo/client'
import { Plus, MessageCircle, Calendar } from 'lucide-react'

const GET_CHATS = gql`
  query GetChats {
    chats(order_by: { updated_at: desc }) {
      id
      title
      created_at
      updated_at
      messages_aggregate {
        aggregate {
          count
        }
      }
    }
  }
`

const CREATE_CHAT = gql`
  mutation CreateChat($title: String!) {
    insert_chats_one(object: { title: $title }) {
      id
      title
      created_at
      updated_at
    }
  }
`

export default function ChatList({ selectedChatId, onSelectChat }) {
  const [isCreating, setIsCreating] = useState(false)
  const [newChatTitle, setNewChatTitle] = useState('')

  const { data, loading, error, refetch } = useQuery(GET_CHATS, {
    pollInterval: 5000, // Poll every 5 seconds for updates
  })

  const [createChat, { loading: creating }] = useMutation(CREATE_CHAT, {
    onCompleted: (data) => {
      onSelectChat(data.insert_chats_one.id)
      setIsCreating(false)
      setNewChatTitle('')
      refetch()
    },
    onError: (error) => {
      console.error('Error creating chat:', error)
    }
  })

  const handleCreateChat = async (e) => {
    e.preventDefault()
    if (!newChatTitle.trim()) return

    try {
      await createChat({
        variables: { title: newChatTitle.trim() }
      })
    } catch (error) {
      console.error('Failed to create chat:', error)
    }
  }

  const formatDate = (dateString) => {
    const date = new Date(dateString)
    const now = new Date()
    const isToday = date.toDateString() === now.toDateString()
    
    if (isToday) {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    } else {
      return date.toLocaleDateString([], { month: 'short', day: 'numeric' })
    }
  }

  if (loading) {
    return (
      <div className="p-4">
        <div className="animate-pulse space-y-3">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="h-16 bg-gray-200 rounded-lg"></div>
          ))}
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-4 text-center">
        <div className="text-red-600 text-sm" data-testid="error-message">
          Error loading chats: {error.message}
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full">
      {/* New Chat Button */}
      <div className="p-4 border-b border-gray-100">
        {!isCreating ? (
          <button
            onClick={() => setIsCreating(true)}
            data-testid="button-new-chat"
            className="w-full flex items-center justify-center space-x-2 p-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
          >
            <Plus className="h-4 w-4" />
            <span>New Chat</span>
          </button>
        ) : (
          <form onSubmit={handleCreateChat} className="space-y-2">
            <input
              type="text"
              value={newChatTitle}
              onChange={(e) => setNewChatTitle(e.target.value)}
              placeholder="Enter chat title..."
              data-testid="input-chat-title"
              className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500"
              autoFocus
            />
            <div className="flex space-x-2">
              <button
                type="submit"
                disabled={creating || !newChatTitle.trim()}
                data-testid="button-create-chat"
                className="flex-1 p-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 disabled:opacity-50 text-sm"
              >
                {creating ? 'Creating...' : 'Create'}
              </button>
              <button
                type="button"
                onClick={() => {
                  setIsCreating(false)
                  setNewChatTitle('')
                }}
                data-testid="button-cancel-chat"
                className="flex-1 p-2 bg-gray-300 text-gray-700 rounded-md hover:bg-gray-400 text-sm"
              >
                Cancel
              </button>
            </div>
          </form>
        )}
      </div>

      {/* Chat List */}
      <div className="flex-1 overflow-y-auto">
        {data?.chats?.length === 0 ? (
          <div className="p-4 text-center">
            <MessageCircle className="h-12 w-12 text-gray-300 mx-auto mb-2" />
            <p className="text-gray-500 text-sm">No chats yet. Create your first chat!</p>
          </div>
        ) : (
          <div className="space-y-1 p-2">
            {data?.chats?.map((chat) => (
              <button
                key={chat.id}
                onClick={() => onSelectChat(chat.id)}
                data-testid={`chat-item-${chat.id}`}
                className={`w-full text-left p-3 rounded-lg transition-colors ${
                  selectedChatId === chat.id
                    ? 'bg-indigo-50 border border-indigo-200'
                    : 'hover:bg-gray-50'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <h3 className="font-medium text-gray-900 truncate" data-testid={`chat-title-${chat.id}`}>
                      {chat.title}
                    </h3>
                    <div className="flex items-center space-x-2 mt-1">
                      <span className="text-xs text-gray-500">
                        {chat.messages_aggregate?.aggregate?.count || 0} messages
                      </span>
                      <span className="text-xs text-gray-400">•</span>
                      <span className="text-xs text-gray-500 flex items-center">
                        <Calendar className="h-3 w-3 mr-1" />
                        {formatDate(chat.updated_at)}
                      </span>
                    </div>
                  </div>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}