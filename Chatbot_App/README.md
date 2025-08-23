# Chatbot Frontend Application

A modern React-based chatbot application with Nhost authentication and real-time GraphQL integration.

## Features

- 🔐 **Email Authentication** - Sign up/Sign in with Nhost Auth
- 💬 **Real-time Chat** - Live message updates via GraphQL subscriptions
- 🤖 **AI Chatbot** - Integration with n8n workflow and OpenRouter API
- 📱 **Responsive Design** - Works on desktop and mobile
- 🎨 **Modern UI** - Clean interface with Tailwind CSS
- 🚀 **Fast Performance** - Built with Vite and optimized for production

## Tech Stack

- **Frontend**: React 18, Vite
- **Authentication**: Nhost Auth
- **Database & API**: Hasura GraphQL, Nhost
- **Styling**: Tailwind CSS
- **Icons**: Lucide React
- **Deployment**: Netlify

## Prerequisites

Before running this application, you need:

1. **Nhost Project** - Create account at [nhost.io](https://nhost.io)
2. **Hasura Database** - Set up tables and permissions
3. **n8n Workflow** - Configure chatbot webhook
4. **OpenRouter API** - For AI responses

## Database Setup

Create these tables in your Hasura console:

### `chats` table

```sql
CREATE TABLE public.chats (
    id uuid DEFAULT gen_random_uuid() NOT NULL PRIMARY KEY,
    title text NOT NULL,
    user_id uuid NOT NULL,
    created_at timestamptz DEFAULT now() NOT NULL,
    updated_at timestamptz DEFAULT now() NOT NULL
);
```

### `messages` table

```sql
CREATE TABLE public.messages (
    id uuid DEFAULT gen_random_uuid() NOT NULL PRIMARY KEY,
    chat_id uuid NOT NULL,
    user_id uuid NOT NULL,
    content text NOT NULL,
    role text NOT NULL CHECK (role IN ('user', 'assistant')),
    created_at timestamptz DEFAULT now() NOT NULL
);
```

### Relationships

- `chats.user_id` → `auth.users.id`
- `messages.chat_id` → `chats.id`
- `messages.user_id` → `auth.users.id`

### Permissions

Set up Row Level Security (RLS) permissions for the `user` role:

**chats table permissions:**

- Insert: `{"user_id": {"_eq": "X-Hasura-User-Id"}}`
- Select: `{"user_id": {"_eq": "X-Hasura-User-Id"}}`
- Update: `{"user_id": {"_eq": "X-Hasura-User-Id"}}`
- Delete: `{"user_id": {"_eq": "X-Hasura-User-Id"}}`

**messages table permissions:**

- Insert: `{"user_id": {"_eq": "X-Hasura-User-Id"}}`
- Select: `{"user_id": {"_eq": "X-Hasura-User-Id"}}`
- Update: `{"user_id": {"_eq": "X-Hasura-User-Id"}}`
- Delete: `{"user_id": {"_eq": "X-Hasura-User-Id"}}`

### Hasura Action

Create a custom action called `sendMessage`:

```graphql
type Mutation {
  sendMessage(chatId: uuid!, message: String!): SendMessageResponse
}

type SendMessageResponse {
  success: Boolean!
  message: String
  response: String
}
```

**Handler URL**: Your n8n webhook URL
**Request Transform**: Forward `chatId` and `message`
**Permissions**: Allow `user` role

## Installation

1. **Clone the repository**

```bash
git clone <your-repo-url>
cd chatbot-frontend
```

2. **Install dependencies**

```bash
npm install
```

3. **Configure environment variables**

```bash
cp .env.example .env
```

Edit `.env` with your Nhost project details:

```env
REACT_APP_NHOST_SUBDOMAIN=your-nhost-subdomain
REACT_APP_NHOST_REGION=your-nhost-region
```

4. **Start development server**

```bash
npm run dev
```

## Deployment to Netlify

### Option 1: Netlify CLI

```bash
npm install -g netlify-cli
netlify build
netlify deploy --prod
```

### Option 2: Git Integration

1. Push your code to GitHub/GitLab
2. Connect repository in Netlify dashboard
3. Set environment variables in Netlify
4. Deploy automatically on push

### Environment Variables on Netlify

Set these in Netlify dashboard under Site Settings > Environment Variables:

- `REACT_APP_NHOST_SUBDOMAIN=your-subdomain`
- `REACT_APP_NHOST_REGION=your-region`

## Project Structure

```
src/
├── components/          # React components
│   ├── Auth.jsx        # Authentication form
│   ├── Dashboard.jsx   # Main app container
│   ├── ChatList.jsx    # Sidebar with chat list
│   ├── ChatView.jsx    # Main chat interface
│   └── Loading.jsx     # Loading component
├── lib/                # Utilities and configuration
│   ├── nhost.js       # Nhost client setup
│   └── graphql.js     # GraphQL queries/mutations
├── App.jsx            # Root component
├── main.jsx          # App entry point
└── index.css         # Global styles
```

## GraphQL Operations

### Queries

- `GET_CHATS` - Fetch user's chat list
- `GET_MESSAGES` - Fetch messages for a chat

### Mutations

- `CREATE_CHAT` - Create new chat
- `SEND_MESSAGE` - Send user message
- `SEND_MESSAGE_ACTION` - Trigger chatbot response
- `DELETE_CHAT` - Delete chat and messages

### Subscriptions

- `MESSAGES_SUBSCRIPTION` - Real-time message updates
- `CHATS_SUBSCRIPTION` - Real-time chat list updates

## n8n Webhook Configuration

Your n8n workflow should:

1. **Receive webhook** from Hasura Action
2. **Validate user ownership** of the chat
3. **Call OpenRouter API** with the message
4. **Save bot response** to database via GraphQL
5. **Return response** to Hasura Action

Example n8n workflow structure:

```
Webhook → Validate User → OpenRouter API → Save to DB → Response
```

## Troubleshooting

### Common Issues

**Authentication not working:**

- Check Nhost subdomain/region in `.env`
- Verify Hasura permissions are set correctly

**Messages not appearing:**

- Check GraphQL subscriptions are enabled
- Verify database table relationships

**Chatbot not responding:**

- Check n8n webhook URL in Hasura Action
- Verify OpenRouter API credentials in n8n

### Development Tips

- Use browser dev tools to inspect GraphQL queries
- Check Hasura console for permission errors
- Monitor n8n execution logs for webhook issues
- Use Netlify deploy logs for build troubleshooting

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For support, please open an issue in the GitHub repository or contact the development team.
