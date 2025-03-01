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

# 02/12/2025

Gemini Flash 1.5 8B spent ~13c on Ages with no examples.
Now I'm trying with 1 training example.

# 02/18/2025

When I provide examples in the user/assistant/user/assistant format (mutiple messages), then the model responds with a classification for each example in the history.

Here is the JSON:

```json
[
  {
    "role": "user",
    "content": [
      {
        "type": "image_url",
        "image_url": {
          "url": "<BASE64>"
        }
      },
      {
        "type": "text",
        "text": "What is this a picture of, western sandpiper, whimbrel, Cooper's hawk, black-bellied plover, sharp-shinned hawk, dunlin, sanderling, Swainson's hawk, semipalmated plover, rough-legged hawk, least sandpiper or bald eagle? Respond with your answer in bold."
      }
    ]
  },
  {
    "role": "assistant",
    "content": "**semipalmated plover**"
  },
  {
    "role": "user",
    "content": [
      {
        "type": "image_url",
        "image_url": {
          "url": "<BASE64>"
        }
      },
      {
        "type": "text",
        "text": "What is this a picture of, least sandpiper, sanderling, whimbrel, dunlin, semipalmated plover, Swainson's hawk, rough-legged hawk, bald eagle, western sandpiper, Cooper's hawk, black-bellied plover or sharp-shinned hawk? Respond with your answer in bold."
      }
    ]
  },
  {
    "role": "assistant",
    "content": "**Cooper's hawk**"
  },
  {
    "role": "user",
    "content": [
      {
        "type": "image_url",
        "image_url": {
          "url": "<BASE64>"
        }
      },
      {
        "type": "text",
        "text": "What is this a picture of, whimbrel, black-bellied plover, least sandpiper, bald eagle, Swainson's hawk, sharp-shinned hawk, dunlin, sanderling, Cooper's hawk, semipalmated plover, rough-legged hawk or western sandpiper? Respond with your answer in bold."
      }
    ]
  },
  {
    "role": "assistant",
    "content": "**bald eagle**"
  },
  {
    "role": "user",
    "content": [
      {
        "type": "image_url",
        "image_url": {
          "url": "<BASE64>"
        }
      },
      {
        "type": "text",
        "text": "What is this a picture of, Swainson's hawk, bald eagle, western sandpiper, whimbrel, black-bellied plover, least sandpiper, semipalmated plover, rough-legged hawk, sanderling, dunlin, sharp-shinned hawk or Cooper's hawk? Respond with your answer in bold."
      }
    ]
  }
]
```

And the model responds with

`"From top to bottom, left to right:\n\n1. **Whimbrel**\n2. **Cooper's hawk**\n3. **Bald eagle**\n4. **Sharp-shinned hawk**"`

# 02/19/2025

[pyinstrument](https://pyinstrument.readthedocs.io/en/latest/guide.html) worked really well for profiling.

Some thoughts on the research questions:

This is sort of a guide for how to apply MLLMs to ai4ecology tasks.
Some of the questions I've had:

* Should I label data? This is the overarching question.
* Do MLLMs do better with more data (few-shot prompting)?
  * Do MLLMs need to be a certain size to leverage more examples?
* Do ViT+kNNs do better?
* How should you include multiple images in MLLM prompts (multiple turns or one message)?

One thing is that I'm not sure how to frame this as a *research* paper.
This is a technical report or a tutorial or a blog post.
But what is the core research question being asked?
Such a question should be timeless to some degree.
Whether to put multiple images in the same message or across multiple turns depends on the current set of MLLMs.
This is likely to change as we continue making better models.
What's the core question?


Timeless Research Framing:

Instead of asking:

    "Do MLLMs outperform traditional ML?"
    → (Too dependent on current model architectures)
    "How do MLLMs scale with data in ecological tasks?"
    → (Still somewhat tied to specifics)

Consider asking:
"What are the fundamental trade-offs between MLLMs and traditional ML for ecological data collection?"

    This frames the problem around trade-offs rather than absolute comparisons, which are bound to change.
    It ensures relevance even as models evolve.
    It guides ecological practitioners on when to choose one approach over another, rather than just evaluating models.

Framing the Contributions as Lasting Principles

Rather than reporting on absolute performance metrics (which will become outdated), the paper could:

    Characterize key factors that determine performance scaling
        E.g., when does task complexity demand more supervision?
        How does domain shift impact MLLMs vs. traditional ML?
        What types of ecological tasks benefit from VLMs (e.g., general object recognition vs fine-grained species classification)?

    Propose a decision-making framework
        How should an ecologist decide between investing in more labeled data vs using an MLLM?
        What characteristics of a dataset (e.g., class imbalance, domain diversity, image quality) affect whether few-shot methods will work?

    Identify inflection points
        Rather than saying "MLLMs need X samples to work," identify the conditions under which adding more data stops improving performance significantly.

Example Hypothesis-Driven Approach

A lasting insight could be testing a hypothesis that generalizes beyond current models, such as:

📌 "Vision-language models rely on semantic generalization, while traditional ML relies on statistical optimization—how does this affect their sample efficiency in ecological tasks?"

    This abstracts away specific model architectures and instead frames the comparison in terms of two fundamental learning paradigms:
        MLLMs (leveraging pre-trained world knowledge, generalizing with few examples)
        Traditional ML (learning purely from the data distribution, scaling better with more labels)
    This would help researchers and practitioners understand the strengths and weaknesses of different approaches even as new models emerge.

Takeaways

If we focus too much on specifics (e.g., "Does GPT-4V outperform a ViT?"), the work will quickly become outdated. Instead:

    Focus on trade-offs rather than head-to-head comparisons.
    Analyze patterns in data efficiency, not just raw performance numbers.
    Frame findings in a way that applies to future models.

Would this kind of framing align with what you're aiming for?



# CVPR Experiment Implementation Checklist

## 1. Models
- [ ] **MLLMs (API-based inference):**
  - [ ] Gemini Flash 2.0
  - [ ] Gemini Flash 1.5 8B
  - [ ] GPT-4o-mini
  - [ ] GPT-4o
  - [ ] Sonnet 3.5
  - [ ] Qwen2-VL 7B
  - [ ] Haiku 3.5
  - [ ] Qwen2-VL 72B
  - [ ] Llama 3.2 11B Vision
  - [ ] Llama 3.2 90B  
  - [ ] **API Settings:** Temperature = 0, batched API calls where possible  

- [ ] **Vision Encoders (Feature Extraction):**
  - [ ] DINOv2 (ViT-B, ViT-L, ViT-H)
  - [ ] CLIP (ViT-B, ViT-L, ViT-H)
  - [ ] SigLIP (ViT-B, ViT-L, ViT-H)
  - [ ] ResNet50 (ImageNet Pretrained)
  - [ ] BioCLIP  

- [ ] **ML Classifiers:**
  - [ ] kNN ($k$ autotuned via GridSearchCV)
  - [ ] SVM (RBF kernel, default hyperparams)
  - [ ] Ridge Classifier (regularization $\alpha=1.0$)

## 2. Datasets & Tasks
- [ ] **Species Classification:**
  - [ ] Birds525
  - [ ] iNat2021
  - [ ] PlantNet
- [ ] **Domain Adaptation:**
  - [ ] Plankton (microscope imagery)
  - [ ] iWildcam (camera trap images)
- [ ] **Functional Trait Prediction:**
  - [ ] FishNet
- [ ] **Generalization:**
  - [ ] Ages (Adult train → Juvenile test)
- [ ] **Multi-task Generalization:**
  - [ ] NeWT Benchmark

## 3. Data Preprocessing
- [ ] Resize images: **Smaller side = 224 px, then center crop to 224×224**
- [ ] Normalize images using **ImageNet mean/std**
- [ ] No augmentations applied (e.g., cropping, flipping)
- [ ] No image modifications before MLLM inference

## 4. Few-shot Sampling & Data Regimes
- [ ] Train subsets: **1, 3, 10, 30, 100, 300, 1000, 3000, 10,000** samples
- [ ] Uniform random sampling from training data
- [ ] **PlantNet & Long-tail datasets:** Secondary experiment with class-balanced sampling
- [ ] Fixed random seed (\texttt{42}) for reproducibility

## 5. MLLM Prompting Strategies
- [ ] **Single-turn prompting:** All few-shot examples + query in a single message
- [ ] **Multi-turn prompting:** Few-shot examples provided sequentially before query
- [ ] **Minimal text prompts:** (e.g., "Classify this species")
- [ ] **CoT variant:** "Think step by step first" appended to prompt (not in few-shot examples)

## 6. MLLM Response Parsing
- [ ] Deterministic regex-based extraction
- [ ] Take the **first species mentioned** if multiple species are listed
- [ ] Track **successful parses** as a function of few-shot examples

## 7. Evaluation Metrics
- [ ] Accuracy@1, Accuracy@5 (micro/macro) for classification tasks
- [ ] Domain adaptation: Accuracy drop (iWildCam, Plankton)
- [ ] Generalization gap (Ages task)
- [ ] FishNet: Mean Squared Error (MSE)
- [ ] **Compute bootstrapped confidence intervals:**
  - [ ] Resample test set **500 times with replacement**
  - [ ] Compute mean metric for each resample
  - [ ] Report **95\% confidence interval**

## 8. Compute Infrastructure
- [ ] **Hardware:**
  - [ ] Traditional ML models trained on NVIDIA A6000 GPUs
  - [ ] ViT-based inference batched for efficiency
  - [ ] MLLM inference run on cloud-based APIs
- [ ] **Parallelization:**
  - [ ] Vision encoder inference parallelized across GPUs
  - [ ] API queries batched where possible

---

## **Final Checks Before Running Experiments**
- [ ] Confirm all models are accessible (API keys, dependencies, etc.)
- [ ] Ensure dataset splits and sampling methods are correctly implemented
- [ ] Validate MLLM output parsing pipeline
- [ ] Test bootstrapping method with a small dataset before full-scale evaluation
- [ ] Set up logging to track MLLM responses, successful parses, and failure cases

---

**Once all boxes are checked, you’re ready to launch experiments!**


I want to summarize the project goals, progress, and next steps.
I need to have a short (2-sentence) and a medium (2-paragraph) description.
A long description is available in this logbook.md and in the paper.

Hi all,

TL;DR: I'm comparing the sample efficiency of MLLMs and CV+ML methods using a suite of biology-related computer vision tasks (https://github.com/samuelstevens/biobench) and making recommendations to practioners in the field.

Why you're receiving this email: I have talked about this project a bunch to many people and I wanted to provide an update. If you aren't interested, just reply saying you don't want to hear about this anymore.

Details: I am evaluating the performance scaling of multimodal large language models (MLLMs) versus traditional vision encoders with machine learning classifiers across diverse ecological tasks, including species classification, domain adaptation, and functional trait prediction using BioBench (https://github.com/samuelstevens/biobench). I am comparing MLLMs (e.g., GPT-4o, Gemini, Qwen2-VL) and vision models (DINOv2, CLIP, SigLIP) under varying data regimes (1–10K samples), prompting strategies (single-turn, multi-turn, CoT), and sampling methods (uniform vs. class-balanced) to determine when additional labeled data improves performance, when MLLMs can compensate for data scarcity, and how prompting choices impact reliability. I expect to find that MLLM performance "saturates" with fewer samples compared to CV+ML methods, and will describe (hopefully) lasting insights based on these empirical findings.

Goals:

1. CV4Animals 4-page submission due March 28th
2. Public codebase and results shortly thereafter

Progress:

1. I have a codebase to evaluate CV+ML methods that is fairly battle-tested.
2. I am developing the MLLMs evaluation methods now.
3. After ICCV (March 7th) I am going to run many experiments in parallel, leveraging API-based MLLMs and local GPUs (OSU, OSC) for CV+ML methods.
4. I have an Overleaf draft with methodological details and a rough outline already. I will continue writing as I get experimental results.

How you can help (if you want):

* Doing analysis. Once I have some experimental results (hopefully March 14th at the latest) then I will want to ask why particular tasks are better/worse, why particular ML methods are better/worse, why particular MLLMs are better/worse, etc. This will need some time with notebooks/scripts to make graphs and tables.
* Writing. The core insight and message is not super clear, even though I feel their is high value in knowing the answer to these questions. Writing the introduction and conclusion would be great.
* Ideas. I would love to hear ideas on models, experiments, tasks or other parts that you are happy to share.

I'm explicitly not asking for help with implementation because I currently wrote the entire codebase  and can hold it all in my head. In combination with AI tool support, I feel that adding additional people to the experiments will slow me down.

I think analysis and meaningful writing will warrant authorship, and offhand ideas warrant acknowledgements. If you feel differently, I am happy to discuss further. Right now, I am first author and Jenna is second author.

Again, if you're not interested, feel free to let me know and I'll take you off this "mailing list"; I plan on sending an update every couple weeks until the CV4Animals deadline.

Best,
Sam

# 02/20/2025

I need to run some experiments with MLLMs and get some progress.

Ages: 1800 test samples (600 x 3)
Plankton: 151K
Newt: 16.4K
iWildcam: 42.7K

Let's just stick with --debug using only 100 random samples.
This is enough for me to notice meaningful differences in prompting strategies (and bugs).

Right now, both models are returning nothing.
(I fixed this by changing the 'role' parameter)

I would like to see one comparison so far:

With 100 randomly sampled examples from `ages`, compare:

* Gemini 1.5 Flash 8B
* Qwen2-VL-7B
* Llama3.2 11B
* GPT-4o-mini

With 0, 1, 3, and 10 samples, multi and single-turn prompts.
I want to know a couple things:

1. How many examples actually fit into their prompt
2. Accuracy (obviously)

Hopefully we will see

1. GPT-4o-mini > Gemini > Qwen > Llama
2. Performance goes up with samples
3. Single > multi

# 02/24/2025

Ages is such a dumb task, it explicitly is a train/test mismatch.
Without any training samples, obviously you do better.

What about FishNet?

FishNet worked! More samples is better.
Now I want to track

1. How many examples actually fit into the prompt.
2. I want to save a sample of the MLLM responses
3. Cost per response

And I need to try multi-turn prompts as well.
Then just show me the freaking graphs baby!

# 02/28/2025

What needs to be done for this benchmark such that I can chill out and work on my ICCV submission?

1. Run bigger MLLMs, on more tasks ($$$)
2. Run more CV models on more tasks (bug with HF)
3. Try different ML models? I think Tanya made the point that if your data is separable, it doesn't matter if you use SVMs, kNNs, linear probing, etc.

Also I would like docs so that others can contribute technical work to this project if they want.

Ok, I got lots of results for two CLIP-trained models (ViT-B/16 and ResNet50) on all NeWT tasks.
Here's what I need to do:

1. Include cluster and subcluster in the schema.sql.
2. Figure out what the deal with the number of test samples in appearance/species.
3. Calculate error bars using means across all tasks in a subcluster, rather than averaging. This means that I probably want to expand the data into a wide list of predictions, then filter and use Polars to efficient processing.
4. Run more CV models, of different sizes and different training procedures.

Questions that I want to understand:

* What do error cases look like? Sample 10-20 images that are incorrectly classified from each task in a subcluster and try to notice any trends.
* Are there obvious trends between model size, model architecture, pretraining type with respect to tasks, sample efficiency and optimal SVM parameters?

These are good questions for Rayeed to dig into, but I shouldn't put him on the critical path for this paper.
