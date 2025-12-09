"""
Generate a sample Q&A dataset for evaluation.

This creates 100 diverse Q&A pairs covering various topics
to test LLM performance across different domains.
"""

import json
import random


def generate_qa_dataset(num_pairs: int = 100) -> list:
    """
    Generate a diverse Q&A dataset.
    
    Args:
        num_pairs: Number of Q&A pairs to generate
        
    Returns:
        List of dictionaries with 'question' and 'reference' keys
    """
    # Pre-defined Q&A pairs covering various topics
    base_qa_pairs = [
        # Science & Technology
        {
            "question": "What is machine learning?",
            "reference": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make predictions or decisions."
        },
        {
            "question": "How does photosynthesis work?",
            "reference": "Photosynthesis is the process by which plants convert light energy into chemical energy. Plants use sunlight, carbon dioxide from the air, and water from the soil to produce glucose (sugar) and oxygen. This process occurs in chloroplasts, primarily in the leaves."
        },
        {
            "question": "What is the difference between DNA and RNA?",
            "reference": "DNA (deoxyribonucleic acid) is a double-stranded molecule that stores genetic information. RNA (ribonucleic acid) is typically single-stranded and plays various roles including protein synthesis. DNA uses thymine, while RNA uses uracil. DNA is found in the nucleus, while RNA can be found in the nucleus and cytoplasm."
        },
        {
            "question": "What is quantum computing?",
            "reference": "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to perform computations. Unlike classical computers that use bits (0 or 1), quantum computers use quantum bits or qubits, which can exist in multiple states simultaneously, potentially solving certain problems much faster."
        },
        {
            "question": "How do vaccines work?",
            "reference": "Vaccines work by introducing a weakened, killed, or partial version of a pathogen into the body. This triggers the immune system to produce antibodies and memory cells. If the person is later exposed to the actual pathogen, their immune system can quickly recognize and fight it off."
        },
        
        # History & Geography
        {
            "question": "What caused World War I?",
            "reference": "World War I was caused by a complex web of factors including militarism, alliances, imperialism, and nationalism. The immediate trigger was the assassination of Archduke Franz Ferdinand of Austria-Hungary in 1914, which led to a chain reaction of declarations of war among European powers."
        },
        {
            "question": "What is the capital of Australia?",
            "reference": "The capital of Australia is Canberra, located in the Australian Capital Territory. It was chosen as the capital in 1908 as a compromise between Sydney and Melbourne, Australia's two largest cities."
        },
        {
            "question": "Who was the first person to walk on the moon?",
            "reference": "Neil Armstrong was the first person to walk on the moon on July 20, 1969, as part of the Apollo 11 mission. He famously said, 'That's one small step for man, one giant leap for mankind.'"
        },
        {
            "question": "What is the Renaissance?",
            "reference": "The Renaissance was a period of cultural, artistic, and intellectual rebirth in Europe from the 14th to the 17th century. It marked a transition from the Middle Ages to modernity, characterized by renewed interest in classical learning, humanism, and artistic innovation."
        },
        {
            "question": "What is the largest ocean?",
            "reference": "The Pacific Ocean is the largest ocean, covering approximately one-third of Earth's surface. It spans from the Arctic in the north to the Antarctic in the south, and from Asia and Australia in the west to the Americas in the east."
        },
        
        # Literature & Arts
        {
            "question": "Who wrote Romeo and Juliet?",
            "reference": "Romeo and Juliet was written by William Shakespeare, an English playwright and poet. It is one of his most famous tragedies, telling the story of two young lovers whose deaths ultimately unite their feuding families."
        },
        {
            "question": "What is the difference between a novel and a novella?",
            "reference": "A novel is a long work of fiction, typically over 40,000 words, with complex plots and multiple characters. A novella is shorter, typically between 17,500 and 40,000 words, with a more focused narrative and fewer characters than a novel."
        },
        {
            "question": "What is impressionism in art?",
            "reference": "Impressionism is an art movement that began in France in the 1860s. It emphasizes capturing the immediate visual impression of a scene, often using visible brushstrokes, natural light, and ordinary subject matter. Famous impressionist artists include Claude Monet and Pierre-Auguste Renoir."
        },
        
        # Mathematics & Logic
        {
            "question": "What is the Pythagorean theorem?",
            "reference": "The Pythagorean theorem states that in a right triangle, the square of the length of the hypotenuse equals the sum of the squares of the lengths of the other two sides. It's expressed as a² + b² = c², where c is the hypotenuse."
        },
        {
            "question": "What is the difference between mean, median, and mode?",
            "reference": "Mean is the average of all numbers (sum divided by count). Median is the middle value when numbers are sorted. Mode is the most frequently occurring value. These are three different measures of central tendency in statistics."
        },
        {
            "question": "What is calculus used for?",
            "reference": "Calculus is used to study rates of change and accumulation. It has applications in physics (motion, forces), engineering (optimization, design), economics (marginal analysis), biology (population growth), and many other fields where change and optimization are important."
        },
        
        # Economics & Business
        {
            "question": "What is inflation?",
            "reference": "Inflation is the rate at which the general level of prices for goods and services rises, eroding purchasing power. It's typically measured as an annual percentage increase. Moderate inflation is normal in growing economies, but high inflation can be problematic."
        },
        {
            "question": "What is the difference between stocks and bonds?",
            "reference": "Stocks represent ownership shares in a company, giving shareholders a claim on assets and earnings. Bonds are debt securities where investors lend money to entities (companies or governments) in exchange for periodic interest payments and return of principal at maturity."
        },
        {
            "question": "What is supply and demand?",
            "reference": "Supply and demand is an economic model that explains how prices are determined in a market. Supply is the quantity of a good producers are willing to sell at various prices. Demand is the quantity consumers are willing to buy. Prices adjust until supply equals demand at equilibrium."
        },
        
        # Health & Medicine
        {
            "question": "What is the difference between a virus and a bacterium?",
            "reference": "Viruses are smaller than bacteria and require a host cell to reproduce. They consist of genetic material (DNA or RNA) in a protein coat. Bacteria are single-celled organisms that can reproduce independently. Antibiotics work against bacteria but not viruses."
        },
        {
            "question": "What is the function of the heart?",
            "reference": "The heart is a muscular organ that pumps blood throughout the body. It has four chambers: two atria (upper) and two ventricles (lower). The right side pumps blood to the lungs for oxygenation, while the left side pumps oxygenated blood to the rest of the body."
        },
        {
            "question": "What is the difference between Type 1 and Type 2 diabetes?",
            "reference": "Type 1 diabetes is an autoimmune condition where the pancreas produces little or no insulin, typically developing in childhood. Type 2 diabetes occurs when the body becomes resistant to insulin or doesn't produce enough, often related to lifestyle factors and typically developing in adulthood."
        },
        
        # Philosophy & Ethics
        {
            "question": "What is the difference between ethics and morality?",
            "reference": "Morality refers to personal or cultural values and principles about right and wrong behavior. Ethics is the philosophical study of morality, examining the nature of moral judgments, values, and principles. Ethics is more theoretical, while morality is more practical."
        },
        {
            "question": "What is utilitarianism?",
            "reference": "Utilitarianism is an ethical theory that holds actions are right if they promote the greatest happiness for the greatest number of people. It's a consequentialist theory, meaning it judges actions by their outcomes rather than their intrinsic nature."
        },
        
        # Psychology
        {
            "question": "What is classical conditioning?",
            "reference": "Classical conditioning is a learning process where a neutral stimulus becomes associated with a meaningful stimulus, eventually eliciting a similar response. Ivan Pavlov's experiment with dogs, where a bell (neutral stimulus) was paired with food (meaningful stimulus), is a famous example."
        },
        {
            "question": "What is the difference between short-term and long-term memory?",
            "reference": "Short-term memory holds information temporarily, typically for seconds to minutes, with limited capacity (about 7±2 items). Long-term memory stores information for extended periods, potentially indefinitely, with much larger capacity. Information moves from short-term to long-term through processes like rehearsal and encoding."
        },
        
        # Environment & Climate
        {
            "question": "What is climate change?",
            "reference": "Climate change refers to long-term changes in global or regional climate patterns, primarily driven by human activities that increase greenhouse gas concentrations in the atmosphere. This leads to rising temperatures, changing precipitation patterns, sea level rise, and more extreme weather events."
        },
        {
            "question": "What is the greenhouse effect?",
            "reference": "The greenhouse effect is a natural process where certain gases in Earth's atmosphere trap heat from the sun, keeping the planet warm enough to support life. However, human activities have increased greenhouse gas concentrations, intensifying this effect and causing global warming."
        },
        {
            "question": "What is biodiversity?",
            "reference": "Biodiversity refers to the variety of life on Earth, including the diversity of species, genetic variation within species, and diversity of ecosystems. High biodiversity is important for ecosystem stability, resilience, and providing ecosystem services that humans depend on."
        },
    ]
    
    # If we need more pairs, we can generate variations or add more
    # For now, we'll cycle through and create variations
    qa_pairs = []
    
    # Use base pairs
    for pair in base_qa_pairs:
        qa_pairs.append(pair)
    
    # Generate variations to reach target number
    while len(qa_pairs) < num_pairs:
        # Create variations by modifying questions slightly
        base_pair = random.choice(base_qa_pairs)
        variations = [
            f"Can you explain {base_pair['question'].lower()}",
            f"Tell me about {base_pair['question'].lower()}",
            f"I'd like to know: {base_pair['question']}",
            base_pair['question'].replace("What is", "Explain what is"),
            base_pair['question'].replace("What", "Can you tell me what"),
        ]
        qa_pairs.append({
            "question": random.choice(variations),
            "reference": base_pair['reference']
        })
    
    # Return exactly num_pairs
    return qa_pairs[:num_pairs]


def save_dataset(qa_pairs: list, filepath: str):
    """Save Q&A dataset to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(qa_pairs, f, indent=2)
    print(f"Dataset saved to {filepath} with {len(qa_pairs)} Q&A pairs")


if __name__ == "__main__":
    # Generate and save dataset
    dataset = generate_qa_dataset(100)
    save_dataset(dataset, "qa_dataset.json")

