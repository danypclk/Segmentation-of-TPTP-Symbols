import os
import wordninja
import re
import matplotlib.pyplot as plt
import json
import enchant
import textwrap

# Initialize the English dictionary
dictionary = enchant.Dict("en_US")

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def analyze_word_segmentation(word_list):
    # Initialize counters and categorized elements for statistics
    stats = {
        'Total Words': 0,
        'Successfully Segmented': 0,
        'Not Segmented': 0,
        'Contains Underscore': 0,
        'Contains CamelStroke': 0,
        'Contains Concatenation': 0,
        'Underscore and Concatenation': 0,
        'CamelStroke and Concatenation': 0,
        'CamelStroke and Underscore': 0,
        'CamelStroke and Underscore and Concatenation': 0,
        'Atomic Words': 0,
        'Uncategorized': 0
    }
    uncategorized_words = []  # List to store uncategorized words
    categorized_elements = {category: [] for category in stats.keys() if category != 'Total Words'}

    for word in word_list:
        contains_camelstroke = False
        contains_underscore = False
        contains_concatenation = False
        stats['Total Words'] += 1  # Count total words processed
        categorized = False  # Track if the word has been categorized

        # Does it have camelcase
        camel_segments = re.findall(r'[a-z]+[A-Z][a-z]+', word)
        if camel_segments:
            contains_camelstroke = True
            categorized = True

        if any(char in "!@#$%^&*()_+-=[]{}|;:'\",.<>?/\\~`" for char in word):
            # Contains Underscore
            contains_underscore = True
            categorized = True

        # Split based on underscores, hyphens, and other non-alphanumeric delimiters
        split_by_symbols = re.split(r'[\W_]+', word)

        # Further split camelCase, PascalCase, and handle capitalized words inside
        words = []
        for token in split_by_symbols:
            if token:
                # Add spaces for camelCase, PascalCase, and sequences of capital letters
                split_camel = re.findall(r'[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+', token)
                words.extend(split_camel)

        # Segmentation attempt
        for w in words:
            segmented_word = wordninja.split(w)
            if len(segmented_word) > 1:
                contains_concatenation = True
                categorized = True
                continue
        
        # Update category counts and categorized elements
        category = None
        if contains_camelstroke:
            if contains_underscore:
                if contains_concatenation:
                    category = 'CamelStroke and Underscore and Concatenation'
                else:
                    category = 'CamelStroke and Underscore'
            elif contains_concatenation:
                category = 'CamelStroke and Concatenation'
            else:
                category = 'Contains CamelStroke'
        elif contains_underscore:
            if contains_concatenation:
                category = 'Underscore and Concatenation'
            else:
                category = 'Contains Underscore'
        elif contains_concatenation:
            category = 'Contains Concatenation'

        if category:
            stats[category] += 1
            categorized_elements[category].append(word)
            stats['Successfully Segmented'] += 1
        else:
            stats['Not Segmented'] += 1
            if dictionary.check(word) and len(word) > 2:
                stats['Atomic Words'] += 1
                category = 'Atomic Words'
                categorized_elements[category].append(word)
            else:
                stats['Uncategorized'] += 1
                uncategorized_words.append(word)
                categorized_elements['Uncategorized'].append(word)

    return stats, uncategorized_words, categorized_elements

def visualize_segmentation_and_features(stats, output_path):
    # Directory to save the charts
    chart_output_dir = os.path.join(output_path, "charts")
    ensure_dir(chart_output_dir)

    # Pie Chart 1: Segmentation Statistics
    segmentation_labels = ["Successfully Segmented", "Not Segmented"]
    segmentation_values = [stats["Successfully Segmented"], stats["Not Segmented"]]

    # Ensure there's data to plot
    if stats["Successfully Segmented"] > 0 or stats["Not Segmented"] > 0:
        plt.figure(figsize=(12, 12))  # Increased figure size
        plt.pie(segmentation_values, labels=segmentation_labels, autopct='%1.1f%%', startangle=140)
        plt.title("Segmentation Results", fontsize=16)
        plt.tight_layout()  # Ensure proper spacing
        plt.savefig(os.path.join(chart_output_dir, "segmentation_results.png"))
        plt.close()
    else:
        print("Segmentation pie chart skipped due to no data available for segmentation results.")

    # Pie Chart 2: Feature-Based Categories
    feature_labels = [
        "Contains Underscore", "Contains CamelStroke",
        "Contains Concatenation", 
        "CamelStroke and Underscore",
        "Underscore and Concatenation",
        "CamelStroke and Concatenation",
        "CamelStroke and Underscore and Concatenation",
        "Atomic Words",
        "Uncategorized"
    ]
    feature_values = [stats[label] for label in feature_labels]

    # Filter out zero-value categories for the pie chart
    non_zero_feature_labels = [label for label, value in zip(feature_labels, feature_values) if value > 0]
    non_zero_feature_values = [value for value in feature_values if value > 0]

    if non_zero_feature_values:  # Ensure there's at least one non-zero value
        plt.figure(figsize=(12, 12))  # Increased figure size
        plt.pie(non_zero_feature_values, labels=non_zero_feature_labels, autopct='%1.1f%%', startangle=140)
        plt.title("Feature-Based Categories", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(chart_output_dir, "feature_based_categories.png"))
        plt.close()
    else:
        print("Feature-based categories pie chart skipped due to no non-zero categories.")

    # Bar Chart: Uncategorized Words
    plt.figure(figsize=(14, 8))  # Increased figure size
    categories = list(stats.keys())[3:]  # Exclude Total Words
    counts = [stats[category] for category in categories]

    plt.bar(categories, counts)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.title("Word Distribution Across Categories", fontsize=16)
    plt.xlabel("Categories", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.tight_layout()  # Adjust layout for better fit
    plt.savefig(os.path.join(chart_output_dir, "word_distribution.png"))
    plt.close()  # Close the figure after saving


def process_json_file(file_path, output_dir):
    with open(file_path, "r") as file:
        data = json.load(file)

    for section in ["predicates", "constants", "variables"]:
        if section in data:
            output_path = os.path.join(output_dir, section)
            ensure_dir(output_path)

            print(f"Processing section: {section}")
            stats, _, categorized_elements = analyze_word_segmentation(data[section])  # Ignore uncategorized_words
            visualize_segmentation_and_features(stats, output_path)

            # Write detailed breakdown to a text file
            breakdown_file_path = os.path.join(output_path, f"{section}_category_breakdown.txt")
            with open(breakdown_file_path, "w") as breakdown_file:
                breakdown_file.write(f"Category Breakdown for {section.capitalize()}\n")
                breakdown_file.write("=" * 40 + "\n")
                for category, elements in categorized_elements.items():
                    breakdown_file.write(f"{category} ({len(elements)} elements):\n")
                    breakdown_file.write("\n".join(elements))
                    breakdown_file.write("\n\n")

# Main script
def main():
    input_dir = "./extracted_data/tptp_library"  # Input directory containing JSON files
    output_dir = "./analysis"  # Output directory for analysis results

    ensure_dir(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(input_dir, file_name)
            print(f"Processing file: {file_name}")
            file_output_dir = os.path.join(output_dir, os.path.splitext(file_name)[0])
            ensure_dir(file_output_dir)
            process_json_file(file_path, file_output_dir)

if __name__ == "__main__":
    main()