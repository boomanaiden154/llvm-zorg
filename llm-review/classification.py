import asyncio
import requests
import json

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

INSTRUCTIONS = """
You are an expert code review agent. Classify the following patch as trivial or
False,t. Your final response should consist solely of a single JSON object of the
following form:

```
{"trivial": <true/false>}
```

You should take into account the following criteria when making a decision:
1. Trivial refactorings that are obviously beneficial and obviously do not
change behavior can be classified as trivial. Otherwise, they should be
considered non-trivial.
2. Test updates can be considered trivial when someone is either adding
test cases or simply regenerating check lines. Otherwise, test updates
should be considered non-trivial.
3. Build system updates like portings to one of the non-standard build
systems such as bazel or gn should generally considered trivial. Other small
build system changes like adding library dependencies or disabling
warnings can also be considered trivial. Otherwise, build system changes
should be marked as non-trivial.
"""

test_commits = {
  "d46cca082b": False,
  "21121533d7": True,
  "2474ac009b": False,
  "7a2aa25c8c": True,
  "868bf3fbb9": True,
  "38949dbc8b": True,
  "75e4aafad0": False,
  "69385549f5": True,
  "f3c6257fab": True,
  "4a32cf0b6d": True,
  "fa46161cbd": True,
  "182ae96a82": False,
  "de3ee84346": True,
  "d09ffba14f": False,
  "00b13e21b6": False,
  "2377f82514": False,
  "a3f12670e6": True,
  "87490ed4ea": True,
  "6beacfbd3f": True,
  "03431a2a68": True,
  "ac4170fe25": False,
  "0646ec9e24": True,
  "2a53990301": True,
  "c93c17b0c0": False,
  "f4ee477c7a": False,
  "176134c603": True,
  "cf80e0eb97": True,
  "20165e1fa0": False,
  "9c1157113e": True,
  "ca27181439": True,
  "1091fd7263": True,
  "d90baf57f3": True,
  "0b806bca3f": True,
  "f97789da65": True,
  "40013b367c": True,
  "c7c289e85c": False,
  "aaf8d4e4da": True,
  "3a1206a755": True,
  "46c1fa8d17": False,
  "856f7d4d33": True,
  "a1476d8c69": True,
  "30e950f4c4": False,
  "195c622972": True,
  "f45f3ce5da": True,
  "6931a33471": True,
  "265dcddad0": True,
}

async def main():
  for commit_sha in test_commits:
  #commit_sha = "5e9f5c5b2b7f05af62c7279a24f91c0b86507d46"
  #commit_sha = "19b19f5ff93c0620b70889e1a5207053861f7f9b"
    print(f"Processing commit {commit_sha}")
    diff_request = requests.get(f"https://github.com/llvm/llvm-project/commit/{commit_sha}.patch")
    diff = diff_request.text

    classifier_agent = LlmAgent(
      name = "classifier",
      description = "Classifies commits",
      model="gemini-3.1-pro-preview",
      instruction=INSTRUCTIONS
    )
    session_service = InMemorySessionService()
    runner = Runner(
      app_name="classifier",
      agent=classifier_agent,
      session_service=session_service
    )
    session = await runner.session_service.create_session(
      app_name="classifier",
      user_id="default_user",
    )
    content = types.Content(
      role = "user",
      parts=[types.Part(text=f"Please review the patch:\n{diff}")]
    )
    print(f"expected: {test_commits[commit_sha]}")
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
        print(f"Actual: {json.loads(message[7:][:-3])['trivial']}")
      #if event.usage_metadata:
      #  print(f"{event.usage_metadata.total_token_count} tokens used")
    print("")


if __name__ == "__main__":
  asyncio.run(main())