import wordninja
import re
import matplotlib.pyplot as plt

def analyze_word_segmentation(word_list):
    # Initialize counters for statistics
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
        'Uncategorized': 0
    }
    uncategorized_words = []  # List to store uncategorized words

    for word in word_list:
        contains_camelstroke = False
        contains_underscore = False
        contains_concatenation = False
        stats['Total Words'] += 1  # Count total words processed
        categorized = False  # Track if the word has been categorized

        # Does it have camelcase
        camel_segments = re.findall(r'[A-Z][a-z]+[A-Z][a-z]+', word)
        if camel_segments:
            contains_camelstroke = True
            categorized = True

        if '_' in word:
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
        
        # Update category counts
        if contains_camelstroke:
            if contains_underscore:
                if contains_concatenation:
                    stats['CamelStroke and Underscore and Concatenation'] += 1
                else:
                    stats['CamelStroke and Underscore'] += 1
            elif contains_concatenation:
                stats['CamelStroke and Concatenation'] += 1
            else:
                stats['Contains CamelStroke'] += 1
        elif contains_underscore:
            if contains_concatenation:
                stats['Underscore and Concatenation'] += 1
            else:
                stats['Contains Underscore'] += 1
        elif contains_concatenation:
            stats['Contains Concatenation'] += 1
        
        # Update segmentation counters
        if categorized:
            stats['Successfully Segmented'] += 1
        else:
            stats['Not Segmented'] += 1
            stats['Uncategorized'] += 1
            uncategorized_words.append(word)  # Add word to uncategorized list
            print(f"Word '{word}' does not fit into any category.")

    return stats, uncategorized_words

def visualize_segmentation_and_features(stats, uncategorized_words):
    # Print uncategorized words
    if uncategorized_words:
        print("\nUncategorized Words:")
        for word in uncategorized_words:
            print(f"  - {word}")
    else:
        print("\nNo uncategorized words found.")

    # Pie Chart 1: Segmentation Statistics
    segmentation_labels = ["Successfully Segmented", "Not Segmented"]
    segmentation_values = [stats["Successfully Segmented"], stats["Not Segmented"]]

    if all(value > 0 for value in segmentation_values):
        plt.figure(figsize=(8, 8))
        plt.pie(segmentation_values, labels=segmentation_labels, autopct='%1.1f%%', startangle=140)
        plt.title("Segmentation Results")
        plt.show()
    else:
        print("Segmentation pie chart skipped due to invalid data.")

    # Pie Chart 2: Feature-Based Categories
    feature_labels = [
        "Contains Underscore", "Contains CamelStroke",
        "Contains Concatenation", 
        "CamelStroke and Underscore",
        "Underscore and Concatenation",
        "CamelStroke and Concatenation",
        "CamelStroke and Underscore and Concatenation",
        "Uncategorized"

    ]
    feature_values = [stats[label] for label in feature_labels]

    # Filter out zero-value categories for the pie chart
    non_zero_feature_labels = [label for label, value in zip(feature_labels, feature_values) if value > 0]
    non_zero_feature_values = [value for value in feature_values if value > 0]

    if non_zero_feature_values:  # Ensure there's at least one non-zero value
        plt.figure(figsize=(8, 8))
        plt.pie(non_zero_feature_values, labels=non_zero_feature_labels, autopct='%1.1f%%', startangle=140)
        plt.title("Feature-Based Categories")
        plt.show()
    else:
        print("Feature-based categories pie chart skipped due to no non-zero categories.")

    # Bar Chart: Uncategorized Words
    plt.figure(figsize=(10, 6))
    categories = list(stats.keys())[3:]  # Exclude Total Words
    counts = [stats[category] for category in categories]

    plt.bar(categories, counts)
    plt.xticks(rotation=45, ha="right")
    plt.title("Word Distribution Across Categories")
    plt.xlabel("Categories")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# Example list of words for testing
words_to_check = [
    "quick_brown_fox", "i_am_a_teapot", "sunny_day_morning",
    "hello_world_test", "multi_word_test", "FourIsstillGood",
    "Example_With_Multi_UnderscoreisindoubtWhat", "IsCamelStroke_GoodTest",
    "ExampleTest_case", "multi_word_Test", "SomeMoreComplexWord",
    "AnotherCamelStrokeExample", "FindTheMiddleGround", "alllowercaseword", "anotherexample",
    "this", "Lets", "randomwordthatfitsnone"
]

# Run the analysis and generate statistics
segmentation_stats, uncategorized_words = analyze_word_segmentation(words_to_check)

# Visualize the segmentation and feature-based categories
visualize_segmentation_and_features(segmentation_stats, uncategorized_words)