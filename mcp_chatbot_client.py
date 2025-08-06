from dotenv import load_dotenv
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import List
import asyncio
import nest_asyncio
import json

nest_asyncio.apply()
load_dotenv()

class MCP_ChatBot:
    def __init__(self):
        """Initialize session and client objects"""
        self.session: ClientSession = None
        self.openai = OpenAI()
        self.available_tools: List[dict] = []
    
    async def process_query(self, query, messages):
        """Process a query using OpenAI with MCP tools"""
        messages.append({'role': 'user', 'content': query})
        
        response = self.openai.chat.completions.create(
            max_tokens=2024,
            model='gpt-3.5-turbo',  # or 'gpt-4o' if you prefer
            tools=self.available_tools,
            messages=messages,
            tool_choice="auto"
        )
        
        process_query = True
        while process_query:
            response_message = response.choices[0].message
            
            # Handle OpenAI's response
            if response_message.content:
                print(response_message.content)
            
            # Check if OpenAI wants to call tools
            if response_message.tool_calls:
                # Add OpenAI's response to conversation
                messages.append(response_message)
                
                # Process each tool call
                for tool_call in response_message.tool_calls:
                    tool_id = tool_call.id
                    tool_args = json.loads(tool_call.function.arguments)
                    tool_name = tool_call.function.name
                    
                    print(f"Calling tool {tool_name} with args {tool_args}")
                    
                    # Call the MCP server tool
                    result = await self.session.call_tool(tool_name, arguments=tool_args)
                    
                    # Send tool result back to OpenAI
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": str(result.content[0].text) if result.content else "No result"
                    })
                
                # Get OpenAI's response to the tool results
                response = self.openai.chat.completions.create(
                    max_tokens=2024,
                    model='gpt-3.5-turbo',
                    tools=self.available_tools,
                    messages=messages,
                    tool_choice="auto"
                )
            else:
                # No more tool calls, we're done
                process_query = False
        
        return messages
    
    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")
        messages = []
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == 'quit':
                    break
                messages = await self.process_query(query, messages)
                print("\n")
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def connect_to_server_and_run(self):
        """Connect to MCP server and run chat loop"""
        # Create server parameters for stdio connection
        server_params = StdioServerParameters(
            command="uv",  # Executable
            args=["run", "mcp_research_server.py"],  # Command line arguments
            env=None  # Optional environment variables
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                self.session = session
                
                # Initialize the connection
                await session.initialize()
                
                # List available tools
                response = await session.list_tools()
                tools = response.tools
                
                print("\nConnected to server with tools:", [tool.name for tool in tools])
                
                # Convert MCP tools to OpenAI format
                self.available_tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema
                        }
                    }
                    for tool in response.tools
                ]
                
                await self.chat_loop()

async def main():
    chatbot = MCP_ChatBot()
    await chatbot.connect_to_server_and_run()

if __name__ == "__main__":
    asyncio.run(main())