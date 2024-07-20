## Note: 
The code provided is optimized for ChatGPT-4. Do not use the free GPT-3.5 and then tell me "ChatGPT is no good" :))

## Example:
[ChatGPT writes code for a new trading strategy](https://chatgpt.com/share/429c8fa1-e84e-4075-8656-d6ddcaf7d73a)

## Template:

### The Template:

Act as a professional programmer who is highly skilled in financial programming, Python, Binance, etc.

Your task: Help me write a new strategy in Python according to the description.

How to do it? Follow these steps:
- Step 1: I will provide an example of my existing implementation for another strategy in my system.
- Step 2: I will give you the description of the new strategy.
- Step 3: You write the code.

IMPORTANT: At each step, you can talk to me to clarify the information until I say "Finish step x" (x=1,2,3). Then we move to the next step. Understand?

### Explanation:  
This template is designed to help you write code more efficiently using ChatGPT-4.

**First line**: "Act as ..." -> Narrow the context. This is crucial as it makes the LLM's answers more focused and significantly better.

**Task**: Clearly define the goal.

**How to do it**: Guide ChatGPT step by step - it's not very good at planning.

**Step 2**: Provide an example - This technique is called _"One-shot prompting"_. By giving it an example, it will produce a much better response.

**Go to the next step only if "Finish step x"**: This technique is an informal form of the _"Chain-of-Thought"_ technique, which guides the LLM and provides/reviews information at each step.

## Extra Information:
If you use GitHub Copilot, after creating several initial strategies/unit tests (or whatever you need to code), use Copilot to help you generate code. You may not need to go back to ChatGPT-4 as often.

However, pay attention: if you want Copilot to generate good code, you have to provide it with "examples" (like the trading class in the template). 
How? Simply open those files/classes in your IDE, and Copilot will read from the open tabs.

# Last Lines:
ChatGPT-4 + Copilot is a very good tool combination. However, it is just a **TOOL** to help you write code faster. The quality depends on you. 
Don't blame the tools.