from arxiv_tools import search_papers, extract_info
import json
from dotenv import load_dotenv
from openai import OpenAI

# MCP tool definitions
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": "Search for papers on arXiv based on a topic and store their information locally",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic to search for"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to retrieve",
                        "default": 5
                    }
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_info",
            "description": "Search for information about a specific paper across all topics",
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "The ID of the paper to look for"
                    }
                },
                "required": ["paper_id"]
            }
        }
    }
]

# Tool execution mapping (will be used by MCP server)
mapping_tool_function = {
    "search_papers": search_papers,
    "extract_info": extract_info
}

def execute_tool(tool_name, tool_args):
    """
    Execute a tool function and format its result appropriately.
    
    Args:
        tool_name: Name of the tool to execute
        tool_args: Dictionary of arguments to pass to the tool
    
    Returns:
        Formatted string result
    """
    result = mapping_tool_function[tool_name](**tool_args)
    
    if result is None:
        result = "The operation completed but didn't return any results."
    elif isinstance(result, list):
        result = ', '.join(result)
    elif isinstance(result, dict):
        # Convert dictionaries to formatted JSON strings
        result = json.dumps(result, indent=2)
    else:
        # For any other type, convert using str()
        result = str(result)
    
    return result


# Initialize OpenAI client
load_dotenv()
client = OpenAI()
tool_keywords = ["paper", "arxiv", "research", "author", "summary", "PDF", "published", "PDF"]

def process_query(query, messages):
    """
    Process a user query using GPT with tool calls as needed, and retained context.
    Args:
        query: The user prompt
        messages: Existing conversation history
    Returns:
        Updated conversation messages
    """
    # Step 1: Start conversation with user query
    messages.append({'role': 'user', 'content': query})
    
    if any(kw in query.lower() for kw in tool_keywords):
        tool_choice = "auto"
    else:
        tool_choice = "none"
    try:
        # Step 2: Send initial request to OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-4o" if you prefer
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,  # Let GPT decide when to use tools
            max_tokens=2024
        )
        
        # Step 3: Process GPT's response in a loop
        process_query = True
        while process_query:
            response_message = response.choices[0].message
            
            # Step 4: Handle GPT's response
            if response_message.content:
                # GPT is speaking normally
                print(response_message.content)
            
            # Check if GPT wants to call tools
            if response_message.tool_calls:
                # Optional: Only allow tool calls for known trigger phrases
                if not any(kw in query.lower() for kw in tool_keywords):
                    print("Skipping tool call because it's not a research-related query.")
                    print("GPT likely misunderstood. Try rephrasing.")
                    return
                # Add GPT's response to conversation
                messages.append(response_message)
                
                # Process each tool call
                for tool_call in response_message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    tool_id = tool_call.id
                    
                    print(f"Calling tool {tool_name} with args {tool_args}")
                    
                    # Execute the tool
                    result = execute_tool(tool_name, tool_args)
                    
                    # Send tool result back to GPT
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": result
                    })
                
                # Get GPT's response to the tool results
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=2024
                )
            else:
                # No more tool calls, we're done
                process_query = False
    except Exception as e:
        print(f"Error inside process_query: {str(e)}")