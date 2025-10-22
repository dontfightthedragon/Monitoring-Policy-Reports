import pandas as pd
import numpy as np

# Load the data
print("Loading data...")
gpt35 = pd.read_csv("/Users/johannahofmann/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Bachelorarbeit/Thesis_rep/analysis/LLM/artifacts/full/llm_predictions_pp_3.5.csv")
gpt4o = pd.read_csv("/Users/johannahofmann/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Bachelorarbeit/Thesis_rep/analysis/LLM/artifacts/full/llm_predictions_pp_4.csv")
gpt35_with_text = pd.read_csv("/Users/johannahofmann/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Bachelorarbeit/Thesis_rep/analysis/LLM/artifacts/full/llm_predictions_w_text.csv")

# Merge on activity_id
df = gpt35.merge(gpt4o, on='activity_id', how='inner', suffixes=('_gpt35', '_gpt4o'))
disagreements = df[df['llm_primary_label_gpt35'] != df['llm_primary_label_gpt4o']]

print(f"Total disagreements: {len(disagreements)}")
print(f"Agreement rate: {(len(df) - len(disagreements)) / len(df) * 100:.2f}%")

# Show disagreement patterns
print("\n=== DISAGREEMENT PATTERNS ===")
disagreement_matrix = pd.crosstab(disagreements['llm_primary_label_gpt35'], 
                                 disagreements['llm_primary_label_gpt4o'])
print(disagreement_matrix)

# Function to show samples
def show_disagreement_samples(gpt35_label, gpt4o_label, n_samples=3):
    pattern = disagreements[
        (disagreements['llm_primary_label_gpt35'] == gpt35_label) & 
        (disagreements['llm_primary_label_gpt4o'] == gpt4o_label)
    ]
    
    if len(pattern) == 0:
        print(f"No samples found for {gpt35_label} -> {gpt4o_label}")
        return
        
    print(f"\n{'='*80}")
    print(f"--- {gpt35_label} (GPT-3.5) → {gpt4o_label} (GPT-4o) [{len(pattern)} cases] ---")
    print(f"{'='*80}")
    
    for i, (_, row) in enumerate(pattern.head(n_samples).iterrows()):
        print(f"\n📋 Sample {i+1} (Activity ID: {row['activity_id']})")
        
        # Get text
        text_row = gpt35_with_text[gpt35_with_text['activity_id'] == row['activity_id']]
        if len(text_row) > 0:
            text = text_row.iloc[0]['text_for_llm']
            if pd.notna(text) and len(str(text)) > 0:
                print(f"📄 Project Text: {str(text)[:500]}{'...' if len(str(text)) > 500 else ''}")
        
        # Show rationales
        if 'llm_rationale_gpt35' in row:
            rationale35 = row['llm_rationale_gpt35']
            if pd.notna(rationale35) and len(str(rationale35)) > 0:
                print(f"🤖 GPT-3.5 rationale: {str(rationale35)[:400]}{'...' if len(str(rationale35)) > 400 else ''}")
        
        if 'llm_rationale_gpt4o' in row:
            rationale4o = row['llm_rationale_gpt4o']
            if pd.notna(rationale4o) and len(str(rationale4o)) > 0:
                print(f"🤖 GPT-4o rationale: {str(rationale4o)[:400]}{'...' if len(str(rationale4o)) > 400 else ''}")
        
        print("-" * 80)

# Show samples for the most common disagreement patterns
print("\n=== SAMPLE DISAGREEMENTS ===")

# Most common patterns based on the matrix
common_patterns = [
    ('FORESTS_LAND', 'CROSS_CUTTING'),  # 2747 cases
    ('WILDLIFE_SPECIES', 'CROSS_CUTTING'),  # 2378 cases  
    ('NONE', 'CROSS_CUTTING'),  # 2050 cases
    ('WATER_MARINE', 'CROSS_CUTTING'),  # 596 cases
    ('CROSS_CUTTING', 'NONE'),  # Need to check this one
    ('FORESTS_LAND', 'NONE'),
    ('WILDLIFE_SPECIES', 'NONE'),
]

for gpt35_label, gpt4o_label in common_patterns:
    show_disagreement_samples(gpt35_label, gpt4o_label, n_samples=2)

# Summary
print(f"\n{'='*80}")
print("=== SUMMARY ===")
print(f"Total disagreements: {len(disagreements):,}")
print(f"Agreement rate: {(len(df) - len(disagreements)) / len(df) * 100:.2f}%")

print("\nGPT-3.5 disagreement labels:")
print(disagreements['llm_primary_label_gpt35'].value_counts())

print("\nGPT-4o disagreement labels:")
print(disagreements['llm_primary_label_gpt4o'].value_counts())

print(f"\nKey insight: GPT-4o is more likely to classify projects as CROSS_CUTTING")
print(f"while GPT-3.5 tends to be more specific (FORESTS_LAND, WILDLIFE_SPECIES, etc.)")
