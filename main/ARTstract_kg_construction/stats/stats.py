import json
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import Counter
import networkx as nx
from matplotlib import cm
from wordcloud import WordCloud
plt.interactive(False)
import math
from wordcloud import WordCloud, get_single_color_func
import matplotlib.pyplot as plt

def load_inputs(ACs_list_name):
    with open(f"../input/{ACs_list_name}.json", "r") as file:
        concept_images = json.load(file)
    with open("../input/merged_ARTstract.json", "r") as file:
        merged_ARTstract = json.load(file)
        return concept_images, merged_ARTstract

def stats_concept_frequencies(dataset_colors, concept_colors, ACs_list_name):

    def concept_frequency_in_source_datasets(ACs_list_name):
        concept_images, merged_ARTstract = load_inputs()
        # Information on the source dataset for each image
        source_datasets = {}
        for image_id, image_info in merged_ARTstract.items(ACs_list_name):
            if "source_dataset" in image_info:
                source_datasets[image_id] = image_info["source_dataset"]

        # Initialize a dictionary to store the frequency of each concept in each source dataset
        concept_frequency = {concept: {dataset: 0 for dataset in set(source_datasets.values())} for concept in concept_images}

        # Count the frequency of each concept in each source dataset
        for concept, images in concept_images.items():
            for image in images:
                dataset = source_datasets.get(image)
                if dataset:
                    concept_frequency[concept][dataset] += 1

        # Print the results
        # for concept, frequencies in concept_frequency.items():
        #     print(f"Concept: {concept}")
        #     for dataset, frequency in frequencies.items():
        #         print(f"  Source Dataset: {dataset}, Frequency: {frequency}")

        return concept_frequency

    def plot_concept_frequencies(concept_frequencies, save_filename, dataset_colors=None, concept_colors=None):
        # Extract the concepts and source datasets
        concepts = list(concept_frequencies.keys())
        source_datasets = set(dataset for frequencies in concept_frequencies.values() for dataset in frequencies.keys())

        # Set up the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.2
        index = list(range(len(concepts)))
        index = np.array(index)


        # Use numpy.arange to calculate the x-coordinates for the bars
        x_coordinates = np.arange(len(concepts))
        concepts = sorted(list(concept_frequencies.keys()))  # Sort concepts alphabetically

        # Sort the source datasets alphabetically to ensure consistent color assignment
        sorted_datasets = sorted(source_datasets)

        # Create a bar for each source dataset
        for i, dataset in enumerate(sorted_datasets):
            frequencies = [concept_frequencies[concept].get(dataset, 0) for concept in concepts]
            color = concept_colors[i % len(concept_colors)] if concept_colors else None  # Use custom colors in a cycle
            ax.bar(x_coordinates + i * bar_width, frequencies, bar_width, label=dataset, color=color)

        # Add labels, title, and legend
        ax.set_xlabel('Concepts')
        ax.set_ylabel('Frequency')
        ax.set_title('Concept Frequencies in Each Source Dataset')
        ax.set_xticks(index + bar_width * (len(source_datasets) - 1) / 2)
        ax.set_xticklabels(concepts)
        ax.legend()

        # Show the plot
        plt.tight_layout()
        plt.show()
        # Save the plot as an image
        plt.savefig(save_filename)

        # Show the plot (optional, you can comment this line out if you don't want to display the plot in the PyCharm's plot viewer)
        plt.show()

    # Get concept frequencies data (format: {concept: {source_dataset: frequency}})
    for ACs_list_name in ACs_list_names:
        concept_frequencies = concept_frequency_in_source_datasets(ACs_list_name)
        # Create data visualizations
        plot_concept_frequencies(concept_frequencies, f'output_imgs/concept_frequencies_{ACs_list_name}.png', dataset_colors=dataset_colors, concept_colors=concept_colors)
    return

def stats_evocation_strengths(dataset_colors, concept_colors, AC_list_names):
    def get_evocation_strength_by_image(ACs_list_name):
        concept_images, merged_ARTstract = load_inputs(ACs_list_name)
        evocation_strength_by_concept = {concept: [] for concept in concept_images}

        for concept, images in concept_images.items():
            for img in images:
                for image_id, image_info in merged_ARTstract.items():
                    if img == image_id:
                        evoked_clusters = merged_ARTstract[img].get("evoked_clusters", {})
                        for cluster_id, cluster_info in evoked_clusters.items():
                            img_evocation_strength = cluster_info.get("evocation_strength")
                        evocation_strength_by_concept[concept].append(img_evocation_strength)

        return evocation_strength_by_concept

    def calculate_average_evocation_strength_by_concept(evocation_strength_by_concept):
        average_strength_by_concept = {}
        for concept, strengths in evocation_strength_by_concept.items():
            if len(strengths) > 0:
                average_strength = sum(strengths) / len(strengths)
                average_strength_by_concept[concept] = average_strength

        return average_strength_by_concept

    def plot_evocation_strengths(evocation_strength_by_concept, save_filename, dataset_colors=None,
                                 concept_colors=None, plot_type="plot"):
        # Extract the concepts and average evocation strengths
        concepts = list(evocation_strength_by_concept.keys())
        average_strengths = [sum(strengths) / len(strengths) if len(strengths) > 0 else 0.0 for strengths in
                             evocation_strength_by_concept.values()]

        # Sort the concepts alphabetically
        sorted_indices = np.argsort(concepts)
        concepts_sorted = [concepts[i] for i in sorted_indices]
        average_strengths_sorted = [average_strengths[i] for i in sorted_indices]

        # Set up the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Use numpy.arange to calculate the x-coordinates for the concepts
        x_coordinates = np.arange(len(concepts))

        # Get the concept colors in the same order as the sorted concepts
        concept_colors_ordered = [concept_colors[i % len(concept_colors)] for i in range(len(concepts_sorted))]

        if plot_type == "plot":
            # Plot the average evocation strengths using a line plot
            ax.plot(x_coordinates, average_strengths_sorted, marker='o', color='b')


        elif plot_type == "bar":
            # Create a bar chart with each concept having its corresponding color
            ax.bar(x_coordinates, average_strengths_sorted, color=concept_colors_ordered)

        # Add labels, title, and legend
        ax.set_xlabel('Concepts')
        ax.set_ylabel('Average Evocation Strength')
        ax.set_title('Average Evocation Strength by Concept')
        ax.set_xticks(x_coordinates)
        ax.set_xticklabels(concepts, rotation=45, ha='right')

        ax.set_ylim(1, 1.5)

        # Show the plot
        plt.tight_layout()
        plt.show()

        # Save the plot as an image
        plt.savefig(save_filename)

        # Show the plot (optional, you can comment this line out if you don't want to display the plot in the PyCharm's plot viewer)
        plt.show()

        return

    for ACs_list_name in ACs_list_names:
        evocation_strength_by_concept = get_evocation_strength_by_image(ACs_list_name)
        average_strength_by_concept = calculate_average_evocation_strength_by_concept(evocation_strength_by_concept)
        plot_type = "plot"
        plot_evocation_strengths(evocation_strength_by_concept, f'output_imgs/{plot_type}_evocation_strengths_{ACs_list_name}.png', dataset_colors=dataset_colors, concept_colors=concept_colors, plot_type=plot_type)

        print(average_strength_by_concept)

def stats_num_detected_objects(dataset_colors, concept_colors, AC_list_names):
    def get_detected_objects_by_image(ACs_list_name):
        concept_images, merged_ARTstract = load_inputs(ACs_list_name)
        detected_objects_by_concept = {concept: [] for concept in concept_images}
        num_detected_objects_by_concept = {concept: [] for concept in concept_images}

        for concept, images in concept_images.items():
            for img in images:
                for image_id, image_info in merged_ARTstract.items():
                    if img == image_id:
                        detected_objects_list = merged_ARTstract[img].get("od", {}).get("ARTstract_od_2023_06_28",
                                                                                        {}).get("detected_objects", [])
                        detected_objects = [detected_object["detected_object"] for detected_object in
                                            detected_objects_list]
                        number_of_detected_objects = len(detected_objects)
                        detected_objects_by_concept[concept].extend(detected_objects)
                        num_detected_objects_by_concept[concept].append(number_of_detected_objects)
        print(detected_objects_by_concept)
        return num_detected_objects_by_concept, detected_objects_by_concept

    def calculate_average_num_detected_objects_by_concept(num_detected_objects_by_concept):
        average_num_detected_objects_by_concept = {}
        for concept, num_detected_objects in num_detected_objects_by_concept.items():
            if len(num_detected_objects) > 0:
                average_num_detected_objects = sum(num_detected_objects) / len(num_detected_objects)
                average_num_detected_objects_by_concept[concept] = average_num_detected_objects

        return average_num_detected_objects_by_concept

    def plot_avg_num_detected_objects(num_detected_objects_by_concept, save_filename, dataset_colors=None,
                                 concept_colors=None, plot_type="plot"):
        # Extract the concepts and average evocation strengths
        concepts = list(num_detected_objects_by_concept.keys())
        average_num_detected_objects = [sum(num_detected_objects) / len(num_detected_objects) if len(num_detected_objects) > 0 else 0.0 for num_detected_objects in
                             num_detected_objects_by_concept.values()]

        # Sort the concepts alphabetically
        sorted_indices = np.argsort(concepts)
        concepts_sorted = [concepts[i] for i in sorted_indices]
        average_num_detected_objects_sorted = [average_num_detected_objects[i] for i in sorted_indices]

        # Set up the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Use numpy.arange to calculate the x-coordinates for the concepts
        x_coordinates = np.arange(len(concepts))

        # Get the concept colors in the same order as the sorted concepts
        concept_colors_ordered = [concept_colors[i % len(concept_colors)] for i in range(len(concepts_sorted))]

        if plot_type == "plot":
            # Plot the average evocation strengths using a line plot
            ax.plot(x_coordinates, average_num_detected_objects_sorted, marker='o', color='b')


        elif plot_type == "bar":
            # Create a bar chart with each concept having its corresponding color
            ax.bar(x_coordinates, average_num_detected_objects_sorted, color=concept_colors_ordered)

        # Add labels, title, and legend
        ax.set_xlabel('Concepts')
        ax.set_ylabel('Average Number of Detected Objects')
        ax.set_title('Average Number of Detected Objects by Concept')
        ax.set_xticks(x_coordinates)
        ax.set_xticklabels(concepts, rotation=45, ha='right')

        ax.set_ylim(1, 5)

        # Show the plot
        plt.tight_layout()
        plt.show()

        # Save the plot as an image
        plt.savefig(save_filename)

        # Show the plot (optional, you can comment this line out if you don't want to display the plot in the PyCharm's plot viewer)
        plt.show()

        return

    for ACs_list_name in ACs_list_names:
        num_detected_objects_by_concept, detected_objects_by_concept = get_detected_objects_by_image(ACs_list_name)
        average_num_detected_objects_by_concept = calculate_average_num_detected_objects_by_concept(num_detected_objects_by_concept)
        plot_type = "plot"
        plot_avg_num_detected_objects(num_detected_objects_by_concept, f'output_imgs/{plot_type}_num_detected_objects_{ACs_list_name}.png', dataset_colors=dataset_colors, concept_colors=concept_colors, plot_type=plot_type)

        print(average_num_detected_objects_by_concept)

def stats_detected_objects(ACs_list_names):
    def get_detected_objects_by_image(ACs_list_name):
        concept_images, merged_ARTstract = load_inputs(ACs_list_name)
        for concept, list in concept_images.items():
            print(concept, "has these many images ", len(list))
        detected_objects_by_concept = {concept: [] for concept in concept_images}
        for concept, images in concept_images.items():
            for img in images:
                for image_id, image_info in merged_ARTstract.items():
                    if img == image_id:
                        detected_objects_list = merged_ARTstract[img].get("od", {}).get("ARTstract_od_2023_06_28",
                                                                                        {}).get("detected_objects",
                                                                                                [])
                        detected_objects = [detected_object["detected_object"] for detected_object in
                                            detected_objects_list]
                        detected_objects_by_concept[concept].extend(detected_objects)
        return detected_objects_by_concept

    def calculate_object_frequencies(detected_objects_by_concept):
        object_frequencies_by_concept = {}
        all_detected_objects = []

        for concept, detected_objects in detected_objects_by_concept.items():
            object_frequencies = Counter(detected_objects)
            object_frequencies_by_concept[concept] = object_frequencies
            all_detected_objects.extend(detected_objects)

        all_object_frequencies = Counter(all_detected_objects)
        return object_frequencies_by_concept, all_object_frequencies

    def plot_object_frequencies(object_frequencies_by_concept):
        num_concepts = len(object_frequencies_by_concept)
        fig, axs = plt.subplots(num_concepts + 1, 1, figsize=(12, 5 * (num_concepts + 1)), sharex=True)

        # Calculate top 30 objects across all concepts, excluding the top object "person"
        all_object_frequencies = {}
        for concept, object_frequencies in object_frequencies_by_concept.items():
            for obj, freq in object_frequencies.items():
                if obj != "person":
                    all_object_frequencies[obj] = all_object_frequencies.get(obj, 0) + freq

        sorted_all_frequencies = sorted(all_object_frequencies.items(), key=lambda x: x[1], reverse=True)
        top_all_objects, top_all_frequencies = zip(*sorted_all_frequencies[:30])

        # Plot top 30 objects across all concepts, excluding the top object "person"
        axs[0].bar(top_all_objects, top_all_frequencies)
        axs[0].set_title("Top 30 Detected Objects Across All Concepts (excluding 'person')")
        axs[0].tick_params(axis='x', rotation=45, labelsize=8)

        for i, (concept, object_frequencies) in enumerate(object_frequencies_by_concept.items(), start=1):
            sorted_frequencies = sorted(object_frequencies.items(), key=lambda x: x[1], reverse=True)
            objects, frequencies = zip(
                *sorted_frequencies[1:31])  # Taking top 30 objects, excluding the first object "person"
            axs[i].bar(objects, frequencies)
            axs[i].set_title(f"Detected Objects for Concept '{concept}'")
            axs[i].tick_params(axis='x', rotation=45, labelsize=8)

        plt.tight_layout()
        plt.show()

        # Save the plot as an image
        save_filename = f"output_imgs/top_30_{ACs_list_name}.jpg"
        plt.savefig(save_filename)

        # Show the plot (optional, you can comment this line out if you don't want to display the plot in the PyCharm's plot viewer)
        plt.show()

    def find_common_objects(object_frequencies_by_concept):
        # Calculate overall frequency of each object across all concepts
        all_object_frequencies = {}
        for object_frequencies in object_frequencies_by_concept.values():
            for obj, freq in object_frequencies.items():
                all_object_frequencies[obj] = all_object_frequencies.get(obj, 0) + freq

        # Sort objects based on their overall frequency in descending order
        sorted_overall_frequencies = sorted(all_object_frequencies.items(), key=lambda x: x[1], reverse=True)
        ordered_objects = [obj for obj, freq in sorted_overall_frequencies]

        # Get the set of common objects that are present in every concept
        common_objects = set.intersection(
            *[set(frequencies.keys()) for frequencies in object_frequencies_by_concept.values()])

        # Return the ordered set of common objects
        ordered_common_objects = [obj for obj in ordered_objects if obj in common_objects]
        # common_objects = find_common_objects(object_frequencies_by_concept)
        print("for all concepts, the most common objects are (from most common to least: ", common_objects)
        return ordered_common_objects

    def find_unique_objects(object_frequencies_by_concept):
        unique_objects_by_concept = {concept: set(frequencies.keys()) for concept, frequencies in
                                     object_frequencies_by_concept.items()}
        for concept, frequencies in object_frequencies_by_concept.items():
            for other_concept, other_frequencies in object_frequencies_by_concept.items():
                if concept != other_concept:
                    unique_objects_by_concept[concept] -= set(other_frequencies.keys())

        for concept, od_set in unique_objects_by_concept.items():
            if len(od_set) == 0:
                print(concept, "is the only one with detected objects: ", od_set)
            else:
                print(concept, "does not have any unique detected object")

        return unique_objects_by_concept

    def find_relevant_objects(object_frequencies_by_concept):
        # Calculate overall frequency of each object across all concepts
        all_object_frequencies = Counter()
        for object_frequencies in object_frequencies_by_concept.values():
            all_object_frequencies.update(object_frequencies)

        relevant_objects_by_concept = {}

        for concept, object_frequencies in object_frequencies_by_concept.items():
            # Calculate relative frequency (TF) for each object within the concept
            relative_frequencies = {
                obj: freq / all_object_frequencies[obj]
                for obj, freq in object_frequencies.items()
            }

            # Calculate inverse concept frequency (IDF) for each object
            num_concepts = len(object_frequencies_by_concept)
            inverse_concept_frequency = {obj: num_concepts / sum(
                1 for concept_freqs in object_frequencies_by_concept.values() if obj in concept_freqs)
                                         for obj in object_frequencies}

            # Calculate relevance score for each object in the concept (TF * IDF)
            relevance_scores = {obj: round(relative_frequencies[obj] * inverse_concept_frequency[obj], 3)
                                for obj in object_frequencies}

            # Sort objects based on their relevance scores in descending order
            sorted_relevance_scores = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)

            # Keep only the objects unique to the current concept's top 10 objects
            top_10_objects = [obj for obj, score in sorted_relevance_scores[:15]]

            # Check if each object appears in only the current concept's top 10 list
            unique_top_10_objects = []
            for obj in top_10_objects:
                unique_to_current_concept = True
                for other_concept, other_top_10 in relevant_objects_by_concept.items():
                    if other_concept != concept and obj in other_top_10:
                        unique_to_current_concept = False
                        break
                if unique_to_current_concept:
                    unique_top_10_objects.append(obj)

            relevant_objects_by_concept[concept] = unique_top_10_objects
        print(relevant_objects_by_concept)
        for concept, od_set in relevant_objects_by_concept.items():
            print(concept, "has relevant concepts: ", od_set)
        return relevant_objects_by_concept

    def find_top_objects(object_frequencies_by_concept, num_top_objects=15):
        top_objects_by_concept = {}
        for concept, frequencies in object_frequencies_by_concept.items():
            total_detected_objects = sum(frequencies.values())
            objects_percentages = {
                obj: (count / total_detected_objects) * 100 for obj, count in frequencies.items()
            }
            sorted_objects = sorted(objects_percentages.items(), key=lambda x: x[1], reverse=True)
            top_objects = dict(sorted_objects[:num_top_objects])
            top_objects_by_concept[concept] = top_objects
        print("Top objects by concept: ", top_objects_by_concept)
        ordered_lists_by_concept = {}
        for concept, object_scores in top_objects_by_concept.items():
            # Sort objects based on their scores in descending order
            sorted_objects = sorted(object_scores.items(), key=lambda x: x[1], reverse=True)
            # Extract the objects from the sorted list
            ordered_objects = [obj for obj, score in sorted_objects]
            ordered_lists_by_concept[concept] = ordered_objects
        for concept, od_set in ordered_lists_by_concept.items():
            print(concept, "has top concepts: ", od_set)
        return top_objects_by_concept
    def find_top_relevant_objects(top_objects_by_concept, relevant_objects_by_concept, k=15):
        top_relevant_objects_by_concept = {}

        for concept, top_concepts in top_objects_by_concept.items():
            # Get the relevant concepts for the current concept
            relevant_concepts = relevant_objects_by_concept.get(concept, [])
            # Calculate the relevance scores for the relevant concepts
            relevance_scores = {obj: 1 for obj in relevant_concepts}
            # Convert top_concepts dictionary to a list of tuples (object, score)
            top_concepts_list = list(top_concepts.items())
            # Sort the top_concepts_list based on scores in descending order
            top_concepts_list.sort(key=lambda x: x[1], reverse=True)
            # Extract the objects from the sorted list
            top_concepts_sorted = [obj for obj, _ in top_concepts_list]
            # Ensure all objects in top_concepts_sorted have scores in relevance_scores dictionary
            relevance_scores.update({obj: 0 for obj in top_concepts_sorted if obj not in relevance_scores})
            # Calculate the Jaccard similarity between the top concepts and relevant concepts
            jaccard_scores = {obj: len(set([obj]).intersection(set(top_concepts_sorted))) / len(
                set([obj]).union(set(top_concepts_sorted))) for obj in relevant_concepts}
            # Sort objects based on Jaccard similarity scores in descending order
            sorted_objects = sorted(jaccard_scores.items(), key=lambda x: x[1], reverse=True)
            # Take only the top k relevant concepts
            top_relevant_objects_by_concept[concept] = [obj for obj, _ in sorted_objects[:k]]
            # Print the top relevant concepts
        for concept, concepts in top_relevant_objects_by_concept.items():
            print(concept, "has top relevant concepts:", concepts)
        print(top_relevant_objects_by_concept)
        return top_relevant_objects_by_concept

    def find_top_relevant_objects_by_concept_w_freqs(top_relevant_objects_by_concept, object_frequencies_by_concept):
        top_relevant_objects_by_concept_w_freqs = {}

        for concept, top_relevant_objects in top_relevant_objects_by_concept.items():
            # Get the object frequencies for the current concept
            object_frequencies = object_frequencies_by_concept.get(concept, {})

            # Fetch frequencies for top relevant objects and create a dictionary {object: frequency}
            objects_with_freqs = {obj: object_frequencies.get(obj, 0) for obj in top_relevant_objects}

            # Store the dictionary {object: frequency} for the current concept
            top_relevant_objects_by_concept_w_freqs[concept] = objects_with_freqs

        # Print the top relevant concepts and their frequencies
        for concept, concepts in top_relevant_objects_by_concept_w_freqs.items():
            print(concept, "has top relevant concepts and frequencies:", concepts)

        return top_relevant_objects_by_concept_w_freqs

    def create_concepts_wordclouds(top_objects_by_concept, top_relevant_objects_by_concept_w_freqs):
        class GroupedColorFunc(object):
            def __init__(self, color_to_words, default_color):
                self.color_func_to_words = [
                    (get_single_color_func(color), set(words))
                    for (color, words) in color_to_words.items()]

                self.default_color_func = get_single_color_func(default_color)

            def get_color_func(self, word):
                """Returns a single_color_func associated with the word"""
                try:
                    color_func = next(
                        color_func for (color_func, words) in self.color_func_to_words
                        if word in words)
                except StopIteration:
                    color_func = self.default_color_func

                return color_func

            def __call__(self, word, **kwargs):
                return self.get_color_func(word)(word, **kwargs)

        font_color = '#0074D9'  # Use any shade of blue you prefer
        helvetica_font = 'Helvetica.ttf'  # Replace with the path to your Helvetica font file

        # Remove "person" from top_objects_by_concept and top_relevant_objects_by_concept
        for top_objects in top_objects_by_concept.values():
            top_objects.pop('person', None)

        for top_relevant_objects in top_relevant_objects_by_concept_w_freqs.values():
            if 'person' in top_relevant_objects:
                top_relevant_objects.remove('person')

        # Generate word clouds for each concept
        for concept, top_objects in top_objects_by_concept.items():
            # Create word cloud objects
            wc_top_objects = WordCloud(
                collocations=False,
                background_color='white',
                font_path=helvetica_font,  # Set the font to Helvetica
                color_func=lambda *args, **kwargs: font_color  # Set all words to blue color
            ).generate_from_frequencies(top_objects)

            # Get the top relevant objects for the current concept
            top_relevant_objects = top_relevant_objects_by_concept_w_freqs.get(concept, {})
            top_relevant_objects = top_relevant_objects_by_concept_w_freqs.get(concept, {})
            wc_top_relevant_objects = WordCloud(
                collocations=False,
                background_color='white',
                font_path=helvetica_font,  # Set the font to Helvetica
                color_func=lambda *args, **kwargs: font_color  # Set all words to blue color
            ).generate_from_frequencies(top_relevant_objects)

            # Plot the word clouds side by side
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            axes[0].imshow(wc_top_objects, interpolation="bilinear")
            axes[0].axis("off")
            axes[0].set_title(f"Top Objects for Concept: {concept}")

            axes[1].imshow(wc_top_relevant_objects, interpolation="bilinear")
            axes[1].axis("off")
            axes[1].set_title(f"Top Relevant Objects for Concept: {concept}")

            plt.show()
            # Save the plot as an image
            save_filename = f"{concept}_{ACs_list_name}_object_wordcloud.jpg"
            plt.savefig(save_filename)
            plt.show()

        return



    for ACs_list_name in ACs_list_names:
        # EXECUTION
        detected_objects_by_concept = get_detected_objects_by_image(ACs_list_name)
        object_frequencies_by_concept, all_object_frequencies = calculate_object_frequencies(
            detected_objects_by_concept)
        plot_object_frequencies(object_frequencies_by_concept)
        relevant_objects_by_concept = find_relevant_objects(object_frequencies_by_concept)
        top_objects_by_concept = find_top_objects(object_frequencies_by_concept)
        top_relevant_objects_by_concept = find_top_relevant_objects(top_objects_by_concept, relevant_objects_by_concept)
        top_relevant_objects_by_concept_w_freqs = find_top_relevant_objects_by_concept_w_freqs(top_relevant_objects_by_concept, object_frequencies_by_concept)
        concepts_wordclouds = create_concepts_wordclouds(top_objects_by_concept, top_relevant_objects_by_concept_w_freqs)

    return



# Execution examples
dataset_colors = ['#00BFFF', '#FF6F61', '#9370DB', '#2E8B57']
concept_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
ACs_list_names = ["Balanced_ARTstract_ACs_lists"]
# ACs_list_names = ["ARTstract_ACs_lists", "Balanced_ARTstract_ACs_lists"]

# Call the functions
# stats_concept_frequencies(dataset_colors, concept_colors, ACs_list_names)
# stats_evocation_strengths(dataset_colors, concept_colors, ACs_list_names)
# stats_num_detected_objects(dataset_colors, concept_colors, ACs_list_names)
# stats_detected_objects(dataset_colors, concept_colors, ACs_list_names)
stats_detected_objects(ACs_list_names)


