import asyncio
import dataclasses
import requests

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

INSTRUCTIONS = """
You an expert code review agent. To ensure that the code diff you are given
meets the appropriate coding standards, please go through the following
checklist and ensure the diff conforms to all the listed points. Do not
comment on anything other than code style. Please record any comments that
you may have using the add_comment tool.

1. Languages and Libraries
  - [ ] C++ Version: Use standard C++17 code; avoid vendor-specific extensions.
  - [ ] Standard Library: Prefer C++ standard library or LLVM support libraries (e.g., ADT) over custom data structures.
  - [ ] LLVM Libraries: Favor LLVM-specific structures (e.g., llvm::DenseMap, llvm::SmallVector) over std::map or std::vector when performance or memory is a concern.
  - [ ] Streams: Use LLVM’s raw_ostream (from Support/raw_ostream.h) instead of std::ostream or <iostream>.
  - [ ] Python: Use Python for automation/scripts. Adhere to PEP 8 and use the Black (v23.x) formatter.

2. Mechanical Source Issues (Formatting)
  - [ ] Golden Rule: If modifying existing code, follow the existing style of the file.
  - [ ] File Headers: Every file must start with the standard license and Doxygen \file header.
  - [ ] Line Width: Limit code to 80 columns.
  - [ ] Indentation: Use spaces (usually 2), never tabs.
  - [ ] Whitespace: No trailing whitespace.
  - [ ] Line Endings: Use Unix (LF) line endings.
  - [ ] Braces:
      - Omit braces for simple, single-statement if/else/loop bodies (unless there's a comment or complex nesting).
      - If one branch of an if/else chain uses braces, all must use them.
  - [ ] Parentheses: Space before ( in control flow (if, for, while), but not in function calls.

3. Commenting and Documentation
  - [ ] Doxygen: Use /// for documentation comments (classes, methods, public APIs).
  - [ ] Class/Method Overviews: Non-trivial classes and methods must have documentation explaining what they do and why.
  - [ ] File Header: Include #ifndef/#define guards using the all-caps path (e.g., LLVM_ANALYSIS_UTILS_LOCAL_H).
  - [ ] C-Style Comments: Only use /* ... */ for inline parameter naming (e.g., foo(/*Prefix=*/nullptr)) or C compatibility.
  - [ ] Diagnostics: Error/warning messages should be lowercase, succinct, and end without a period.

4. #include Style
  - [ ] Order:
      1. Main Module Header (the header for the current .cpp file).
      2. Local/Private Headers.
      3. LLVM project/subproject headers (clang/..., llvm/...).
      4. System #includes.
  - [ ] Sorting: Sort each group lexicographically.
  - [ ] Minimalism: Include as little as possible. Use forward declarations instead of headers where possible.

5. Language and Compiler Issues
  - [ ] Warnings: Treat all compiler warnings as errors.
  - [ ] RTTI & Exceptions: Do not use RTTI (dynamic_cast) or C++ Exceptions. Use LLVM's isa<>, cast<>, and dyn_cast<> instead.
  - [ ] Casts: Prefer C++-style casts (static_cast, etc.) over C-style casts.
  - [ ] Static Constructors: Do not use global constructors or destructors.
  - [ ] class vs struct: Use struct only when all members are public.
  - [ ] auto: Use auto only when it improves readability (e.g., with cast<T>, or complex iterators). Always use auto * or auto & to avoid accidental copies.
  - [ ] std::sort: Use llvm::sort instead to ensure deterministic behavior.

6. Design and Style
  - [ ] Naming Conventions:
      - Types/Variables: CamelCase (start with UpperCase).
      - Functions: camelCase (start with lowercase).
      - Enumerators: VK_Prefix (usually Uppercase with prefix).
  - [ ] Early Exits: Use early return or continue to reduce nesting and simplify logic.
  - [ ] Namespace Polluting: Never use using namespace std;. Use using namespace llvm; only in .cpp files.
  - [ ] Assertions: Assert liberally using assert(condition && "Message"). Use llvm_unreachable("Message") for dead code paths.
  - [ ] Virtual Methods: If a class has virtual methods, provide at least one out-of-line virtual method "anchor" in the .cpp file to prevent vtable bloating.
  - [ ] Loops: Prefer range-based for loops over explicit iterators. If using iterators, don't re-evaluate .end() every iteration.
  - [ ] Visibility: Keep headers self-contained. Use anonymous namespaces or static for file-local functions/variables.
"""

@dataclasses.dataclass
class ReviewComment:
  comment_text: str
  file_name: str
  line_number: int

def get_add_comment_tool(comments_list: list[ReviewComment]):
  def add_comment(comment_text: str, file_name: str, line_number: int):
    comments_list.append(ReviewComment(comment_text, file_name, line_number))

  return add_comment

async def main():
  #PR_NUMBER = 198411
  #PR_NUMBER = 197850
  #diff_request = requests.get(f"https://patch-diff.githubusercontent.com/raw/llvm/llvm-project/pull/{PR_NUMBER}.diff")
  #diff = diff_request.text
  diff = open("/tmp/test.patch").read()

  comments_list = []
  style_review = LlmAgent(
    name = "style_review",
    description = "Reviews code style",
    model="gemini-3.1-pro-preview",
    instruction=INSTRUCTIONS,
    tools=[get_add_comment_tool(comments_list)]
  )
  session_service = InMemorySessionService()
  runner = Runner(
    app_name="style_review",
    agent=style_review,
    session_service=session_service
  )
  session = await runner.session_service.create_session(
    app_name="style_review",
    user_id="default_user",
  )
  content = types.Content(
    role="user",
    parts=[types.Part(text=f"Please review the diff:\n{diff}")]
  )
  async for event in runner.run_async(
    user_id="default_user", session_id=session.id, new_message=content
  ):
    if event.get_function_calls():
      print(f"Function Calls: {[fn_call.name for fn_call in event.get_function_calls()]}")
    elif event.get_function_responses():
      print(f"Function Responses: {[fn_resp.name for fn_resp in event.get_function_responses()]}")
    elif event.content and event.content.parts:
      message = "\n\n".join(
          part.text for part in event.content.parts if part.text
      )
      print(message)
    if event.usage_metadata:
      print(f"{event.usage_metadata.total_token_count} tokens used")
  print(comments_list)

if __name__ == "__main__":
  asyncio.run(main())