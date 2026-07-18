# MaxShapley Annotation Guide

## Introduction

Welcome to the MaxShapley annotation project! This guide will help you understand how to annotate source relevance for multi-hop question answering datasets.

### What are we annotating?

We're evaluating how relevant each source (context passage) is to answering a given question. This helps us understand which pieces of information are essential for answering complex, multi-hop questions.

### Why is this important?

Your annotations will be used to:
- Evaluate source attribution methods in multi-hop QA systems
- Understand what makes sources relevant in complex reasoning tasks
- Train and evaluate AI systems that need to identify relevant information

## The Relevance Scale

You'll rate each source on a scale from 0 to 3:

### 3 - Highly Relevant
**Key information directly needed to answer the question**

- Contains facts, entities, or relationships essential to the answer
- Removing this source would make it impossible or very difficult to answer
- In multi-hop questions, this includes sources that provide critical reasoning steps

**Example:** For "What genre of music does the artist who plays piano on santoor perform?"
- A source describing the artist's music genre would be rated 3
- A source defining santoor as a trapezoid-shaped instrument would be rated 3

### 2 - Moderately Relevant
**Provides supporting context or partial information**

- Contains information that helps understand the question or answer
- Provides background context that supports the reasoning
- Contains related information but not the key facts needed

**Example:**
- A source describing the history of the santoor instrument
- A source about the artist's other musical instruments

### 1 - Weakly Relevant
**Tangentially related, provides background but not critical**

- Mentions entities from the question but doesn't help answer it
- Provides very general background information
- Related topic but different focus

**Example:**
- A source about a different artist who also plays santoor
- A source about the cultural origins of similar instruments

### 0 - Not Relevant
**No useful information for answering the question**

- Completely unrelated to the question
- Mentions the same words but in a different context
- Cannot contribute to answering the question in any meaningful way

**Example:**
- A source about a completely different instrument or artist
- A source that happens to mention "music" but is about something else

## Annotation Guidelines

### Before You Start

1. **Read the question carefully** - Make sure you understand what's being asked
2. **Review the answer if provided** - This helps you know what information would be relevant
3. **Consider the question type:**
   - **Single-hop:** Usually requires information from 1-2 sources
   - **Multi-hop:** Requires connecting information from 2+ sources

### While Annotating

1. **Evaluate each source independently** - Don't worry about how many 3's or 0's you give
2. **Focus on utility, not just topic overlap** - A source can mention the same entities but still be irrelevant
3. **Consider completeness:**
   - Does this source alone answer the question? → Likely 3
   - Does it provide part of the answer? → Likely 2 or 3
   - Does it just provide context? → Likely 1 or 2
   - Is it unhelpful? → Likely 0

4. **For multi-hop questions:**
   - Multiple sources can be rated 3 if they each provide essential reasoning steps
   - Sources that connect the reasoning chain are typically highly relevant

5. **When uncertain between two scores, choose the lower one** - Be conservative

### Special Cases

#### MS MARCO Dataset
- No ground truth answer is provided
- Judge based on what information would likely answer the question
- Consider what a search engine user would find helpful

#### Conflicting Information
- If a source contains incorrect information, rate based on: would it be useful if it were correct?
- Annotate based on relevance to the question, not factual accuracy

#### Incomplete Information
- If a source provides partial but useful information, rate it 2 (moderate) or 3 (high) depending on how essential that partial information is

## Examples

### Example 1: HotpotQA

**Question:** "What genre of music is Adnan Sami noted for playing on the piano created through a trapezoid-shaped hammered dulcimer?"

**Answer:** "Indian classical music"

**Sources:**

1. **Adnan Sami** - "Adnan Sami Khan is an Indian singer, musician, music composer, pianist and actor. He performs Indian and western music... He is noted for playing Indian classical music on the piano created through the Santoor."
   - **Rating: 3 (Highly Relevant)** - Directly answers the question about the genre

2. **Santoor** - "The santoor is a trapezoid-shaped hammered dulcimer or string musical instrument..."
   - **Rating: 3 (Highly Relevant)** - Confirms santoor is a trapezoid-shaped hammered dulcimer, essential for the question

3. **Khim** - "The Khim is a stringed musical instrument that is from Persia, called Hammered Dulcimer or Cimbalon..."
   - **Rating: 1 (Weakly Relevant)** - About a similar instrument but not the one in question

4. **Jim Couza** - "Jim Couza was an American hammered dulcimer player..."
   - **Rating: 0 (Not Relevant)** - Different person, different instrument, doesn't help answer

### Example 2: MS MARCO

**Question:** "definition of a sigmet"

**Sources:**

1. **985989** - "The criteria for a non-convective SIGMET to be issued are severe or greater turbulence..."
   - **Rating: 3 (Highly Relevant)** - Describes what SIGMET is and criteria for issuance

2. **7707868** - "Information available includes icing, turbulence, convection, PIREP, METAR, TAF, AIRMET, SIGMET..."
   - **Rating: 1 (Weakly Relevant)** - Mentions SIGMET in a list but doesn't define it

3. **3299960** - "Definition 1. A point is that which has no part. Definition 2. A line is breadthless length..."
   - **Rating: 0 (Not Relevant)** - About geometric definitions, completely unrelated

### Example 3: Multi-hop Reasoning

**Question:** "In which country was the director of the 2010 film The Social Network born?"

**Answer:** "United States"

**Sources:**

1. **The Social Network** - "The Social Network is a 2010 film directed by David Fincher..."
   - **Rating: 3 (Highly Relevant)** - First hop: identifies the director

2. **David Fincher** - "David Fincher is an American film director born in Denver, Colorado, United States..."
   - **Rating: 3 (Highly Relevant)** - Second hop: provides birth country

3. **Mark Zuckerberg** - "Mark Zuckerberg is the founder of Facebook, which the film depicts..."
   - **Rating: 1 (Weakly Relevant)** - Related to the film but doesn't help answer the question

## Best Practices

### Quality Over Speed
- Take your time to read each source carefully
- It's better to annotate 10 samples well than 30 samples carelessly
- Aim for 10-15 samples per annotation session to avoid fatigue

### Consistency
- Try to apply the same standards across all samples
- If you're unsure, refer back to this guide and the examples
- Remember your previous annotations and try to be consistent

### When to Take a Break
- If you find yourself rushing through sources without reading carefully
- If you're having trouble concentrating
- After 30-45 minutes of continuous annotation

### Communication
- If you encounter confusing samples, note them down
- If you're unsure about edge cases, ask for clarification
- Share interesting patterns you notice

## Frequently Asked Questions

**Q: How many sources should be rated 3?**
A: There's no fixed number. Some questions might have 1 highly relevant source, others might have 3-4 for multi-hop reasoning.

**Q: What if I disagree with the provided answer?**
A: Annotate based on the provided answer. If you strongly believe it's wrong, make a note.

**Q: Can a source be rated 3 if it doesn't contain the final answer?**
A: Yes! In multi-hop questions, sources that provide essential intermediate steps should be rated 3.

**Q: What if the source is very long and only one sentence is relevant?**
A: Rate based on whether the relevant information is there, regardless of the source length.

**Q: What if I make a mistake?**
A: You can go back to any sample and re-annotate it. The tool allows you to modify or delete your annotations.

## Getting Help

If you have questions or run into issues:
1. Review this guide and the examples
2. Check the annotation tool's built-in help (sidebar)
3. Consult with the project coordinator
4. Look at the tool's README: `annotation_tool/README.md`

## Summary Checklist

Before saving each annotation, ask yourself:

- [ ] Did I read the question carefully?
- [ ] Did I read each source completely?
- [ ] Did I consider what information is needed to answer the question?
- [ ] Did I evaluate each source independently?
- [ ] Am I confident in my ratings?
- [ ] Did I annotate all sources?

Thank you for your careful and thoughtful annotations!
