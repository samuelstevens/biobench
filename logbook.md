# Logbook

This provides a set of notes, step-by-step, of my process developing this paper.
In combination with the preprint, code comments, and git blame, it's probably the best way to understand *why* a decision was made, rather than *what* decision was made.

*Open science y'all*

# 02/10/2025

What are the core experiments?

We need to choose different number of samples, and run experiments multiple times.
I think I have a good set of tasks.
NeWT is probably the best first task.
Then we sample 0, 1, 3, 10, 30, 100, 300, ... MAX samples per task.
For each task and number of samples, we need to run at least 5 trials.
I should double check and see how much it would cost me to run these trials with OpenAI's GPT-4o mini.

GPT-4o mini:
Affordable small model for fast, everyday tasks | 128k context length
* Input: $0.150 / 1M tokens
* Cached input: $0.075 / 1M tokens
* Output: $0.600 / 1M tokens

Plus I can halve these costs with the Batched API.

Sonnet 3.5 is $3M / 1M input tokens, which is waaaay more expensive.
Even Haiku is $0.80 / 1M tokens, which is like 6 times more expensive.
Sorry Anthropic :(

Gemini 2.0 is $0.10 / 1M input tokens and $0.40 / 1M output tokens, which is 1/3 cheaper than 4o mini.
Then I could try Llama 3.2 11B vision which I know is garbage, and Pixtral 12B.
I think that four models is enough.
If moondream is available, that would be great.

I can do CLIP, BioCLIP, DINOv2 and ResNet50 for vision models.

How would I get this result as soon as possible?

# 02/11/2025

Some paper motivtion ideas from Claude:

Goal: Enable informed data collection strategies for ecological ML systems
Problem: Ecologists are willing to invest in data labeling, but lack clear guidance on whether additional labeling effort will translate to meaningful performance improvements
Solution: Empirical analysis of performance scaling curves for VLMs vs traditional ML approaches, revealing the point at which additional data collection yields diminishing returns

Goal: Optimize resource allocation in ecological ML projects
Problem: Projects must decide upfront whether to invest in (a) extensive data collection and traditional ML or (b) minimal data collection and VLMs, but lack empirical guidance for this decision
Solution: Characterization of performance vs sample size curves across multiple ecological tasks, enabling data-driven decisions about collection strategies

Goal: Bridge the "medium-data" gap in ecological ML
Problem: Current literature focuses on few-shot (1-5 samples) or large-data (10000+ samples) regimes, while many ecological projects operate in the 100-1000 sample range
Solution: Targeted analysis of the critical "medium-data" regime where VLM and traditional ML approaches may cross over in performance

Goal: Create sustainable ecological ML pipelines
Problem: Projects often start with few samples but accumulate more over time, making it unclear whether to invest in VLMs (good initial performance) or traditional ML (better scaling)
Solution: Framework for understanding when and how to transition between approaches as more data becomes available

Goal: Align ML approaches with ecological data collection realities
Problem: Standard few-shot learning research assumes fixed, small sample sizes, while ecological projects often have gradually expanding datasets
Solution: Analysis of how different approaches scale with increasing data, matching real-world ecological data collection patterns
