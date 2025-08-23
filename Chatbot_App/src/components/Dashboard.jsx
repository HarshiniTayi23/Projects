import { useState } from 'react'
import { useSignOut, useUserData } from '@nhost/react'
import { LogOut, Plus, MessageCircle } from 'lucide-react'
import ChatList from './ChatList'
import ChatView from './ChatView'

export default function Dashboard() {
  const [selectedChatId, setSelectedChatId] = useState(null)
  const { signOut } = useSignOut()
  const user = useUserData()

  const handleSignOut = () => {
    signOut()
  }

  return (
    <div className="h-screen bg-gray-100 flex">
      {/* Sidebar */}
      <div className="w-80 bg-white border-r border-gray-200 flex flex-col">
        {/* Header */}
        <div className="p-4 border-b border-gray-200">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-2">
              <MessageCircle className="h-6 w-6 text-indigo-600" />
              <h1 className="text-xl font-semibold text-gray-900">ChatBot</h1>
            </div>
            <button
              onClick={handleSignOut}
              data-testid="button-signout"
              className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-md"
              title="Sign out"
            >
              <LogOut className="h-5 w-5" />
            </button>
          </div>
          
          <div className="text-sm text-gray-600" data-testid="user-info">
            Welcome, {user?.email}
          </div>
        </div>

        {/* Chat List */}
        <div className="flex-1 overflow-hidden">
          <ChatList 
            selectedChatId={selectedChatId}
            onSelectChat={setSelectedChatId}
          />
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {selectedChatId ? (
          <ChatView chatId={selectedChatId} />
        ) : (
          <div className="flex-1 flex items-center justify-center bg-gray-50">
            <div className="text-center">
              <MessageCircle className="h-16 w-16 text-gray-300 mx-auto mb-4" />
              <h2 className="text-xl font-medium text-gray-900 mb-2">Welcome to ChatBot</h2>
              <p className="text-gray-600 mb-6">Select a chat to start messaging or create a new one</p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}