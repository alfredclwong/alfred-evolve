ISLAND_VARIATIONS = [
    ["", "rewrite"],
    ["explore", "plan", "refactor"],
    ["tweak", "optimize", "simplify"],
]

VARIATIONS = {
    "": "",
    "tweak": (
        "Try to tweak the parameters used in the previous completions to improve the score. "
        "This might include changing weights, adjusting learning rates, or modifying other hyperparameters. "
        "The goal is to find a better configuration that leads to a higher score without changing the "
        "overall approach or algorithm significantly."
    ),
    "explore": (
        "The iterative process has reached a plateau, so we need to explore new ideas to make progress. "
        "Try to explore new approaches or techniques that were not used in the previous completions. "
        "This might include trying different algorithms, using new libraries, or implementing novel "
        "data structures. The goal is to break out of the current pattern and find a new direction that "
        "could lead to a higher score. This could involve significant changes to the code or approach, "
        "so be creative and think outside the box."
    ),
    "simplify": (
        "Try to simplify the previous completions by removing unnecessary complexity or "
        "redundancy. This might include eliminating unused variables, reducing the number of "
        "functions, or streamlining the logic. The goal is to make the code more straightforward "
        "and easier to understand, which could lead to a higher score. Simplification can also "
        "improve performance by reducing the amount of code that needs to be executed."
    ),
    "refactor": (
        "Try to refactor the previous completions by improving the structure or organization of the code. "
        "This might include breaking down large functions into smaller ones, improving variable names, "
        "or reorganizing the code for better readability. The goal is to make the code cleaner and more maintainable, "
        "which could lead to a higher score."
    ),
    "optimize": (
        "Try to optimize the previous completions by improving performance or reducing resource usage. "
        "This might include optimizing algorithms, reducing memory usage, or improving execution speed. "
        "The goal is to make the code more efficient and effective, which could lead to a higher score."
    ),
    "plan": (
        "Some ideas may take several iterations to implement, so it is important to plan ahead. "
        "Try to outline a plan for how to implement the next steps in the evolution process. "
        "This might include identifying key areas to focus on, setting specific goals for the next "
        "completions, or outlining a strategy for how to approach the task. The goal is to have a clear "
        "direction for the next steps, which could lead to a higher score."
    ),
    "rewrite": (
        "The previous completions have not made significant progress, so it is time to rewrite the code. "
        "Use a SEARCH/REPLACE block with an empty SEARCH section to replace the entire parent program "
        "with a new implementation. Don't be afraid to start from scratch and try a completely new approach. "
        "The goal is to create a fresh implementation that could lead to a higher score, even if it means "
        "discarding previous work. This could involve significant changes to the code or approach, "
        "so be creative and think outside the box."
    ),
}

PREMABLE = """\
Act as an expert Python developer. Your job is to make iterative improvements \
to a source file in order to score highly on a task. You will be provided with \
a task description, a parent program, and a set of inspiration programs, which \
are previous attempts at solving the task. Your output will be a diff that \
will be applied to the parent program to create a new program.\
"""
EPILOGUE = """\
Your output should consist of two parts: your reasoning for the changes and \
the diff itself. The reasoning should be a concise bullet-point list of the \
reasons why you believe the diff will improve the program's score. The diff \
should consist of SEARCH/REPLACE blocks that can be applied to the parent, and \
no other text. One diff may contain multiple SEARCH/REPLACE blocks, separated \
by newlines. The resulting program should be a valid Python program that will \
attempt to solve the task. It is important that the diff and code are valid, \
as invalid outputs will waste resources and time. Your response is limited to \
a maximum of 8192 tokens, so your changes must be small and focused.

SEARCH/REPLACE block rules:
1. Each SEARCH/REPLACE block consists of a SEARCH section and a REPLACE section
2. The SEARCH section begins with `<<<<<<<< SEARCH`, the REPLACE section \
begins with `========`, and the end of the block is marked with `>>>>>>>> \
REPLACE`.
3. The SEARCH section contains the code to be replaced, which should be \
uniquely identifiable within the parent program.
4. The REPLACE section contains the code that should replace the SEARCH code.
5. Both sections operate on a line-by-line basis.
6. A special case is when the SEARCH section is empty, which means the entire \
parent program should be replaced with the REPLACE section.

Example #1:
<PARENT>
def main():
    print("Hello, world!")
</PARENT>
<DIFF>
<<<<<<<< SEARCH
    print("Hello, world!")
========
    print("Aloha, world!")
>>>>>>>> REPLACE
</DIFF>

Example #2:
<PARENT>
def main():
    print("Hello, world!")
</PARENT>
<DIFF>
<<<<<<<< SEARCH
========
if __name__ == "__main__":
    print("Aloha, world!")
>>>>>>>> REPLACE

Your output should be formatted as follows:

<REASONING>{reasoning}</REASONING>
<DIFF>{diff}</DIFF>
"""
