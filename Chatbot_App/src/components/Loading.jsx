import { MessageCircle } from 'lucide-react'

export default function Loading() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
      <div className="text-center">
        <div className="mx-auto h-12 w-12 bg-indigo-600 rounded-full flex items-center justify-center mb-4 animate-pulse">
          <MessageCircle className="h-6 w-6 text-white" />
        </div>
        <div className="text-indigo-600 text-lg font-medium" data-testid="loading-message">
          Loading...
        </div>
      </div>
    </div>
  )
}