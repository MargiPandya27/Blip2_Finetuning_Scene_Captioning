import json
from collections import defaultdict
import os
import pandas as pd # <-- Import the pandas library



def generate_raw_caption(frame_data):
    """
    Generates a raw, descriptive natural language caption from a single BDD-100K frame's JSON data.
    """

    # 1. Extract Global Attributes (BDD-100K top-level keys)
    time_of_day = frame_data.get('attributes', {}).get('timeofday', 'unknown time of day')
    weather = frame_data.get('attributes', {}).get('weather', 'unknown weather')
    scene = frame_data.get('attributes', {}).get('scene', 'unknown scene')

    # Base sentence structure for the environment
    caption_parts = []
    caption_parts.append(f"It is a **{time_of_day}** scene in **{weather}** weather on a **{scene}** road.")

    # 2. Aggregate Object Counts and Critical Attributes
    object_counts = defaultdict(int)
    traffic_light_status = None

    for obj in frame_data.get('labels', []):
        category = obj.get('category')
        attributes = obj.get('attributes', {})

        # Capture traffic light color separately as it's critical
        if category == 'traffic light' and attributes.get('trafficLightColor') != 'none':
            # Prioritize the most restrictive (non-green/none) color if multiple are present
            color = attributes.get('trafficLightColor')
            if color in ['red', 'yellow']:
                 traffic_light_status = color
            elif traffic_light_status is None and color == 'green':
                 traffic_light_status = color

        # General object count, appending attributes if they are important (occluded/truncated)
        if category and category != 'traffic light':
            descriptor = ""
            if attributes.get('occluded'):
                descriptor += "partially occluded "
            if attributes.get('truncated'):
                descriptor += "truncated "

            # Use a combined key (e.g., "car" or "partially occluded car") for counting
            key = (descriptor + category).strip()
            object_counts[key] += 1

    # 3. Incorporate Object Information
    object_descriptions = []

    # Traffic light is the highest priority object
    if traffic_light_status:
        object_descriptions.append(f"The traffic light is **{traffic_light_status}**.")

    # Add general objects
    for item, count in sorted(object_counts.items()):
        if count == 1:
            object_descriptions.append(f"There is one **{item}**.")
        else:
            object_descriptions.append(f"There are **{count} {item}s**.")

    # 4. Final Concatenation
    if object_descriptions:
        object_sentence = " ".join(object_descriptions)
        caption_parts.append(f"Key visible objects are: {object_sentence}")

    return " ".join(caption_parts)


def process_bdd_labels(file_path):
    """
    Main function to load the BDD-100K JSON and process all frames.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return [] # Return empty list on error

    print(f"Loading data from: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print("Error: Could not parse JSON file.")
        return [] # Return empty list on error

    processed_data = []

    # Loop through all entries
    for frame in data:
        # Generate the structured caption
        raw_caption = generate_raw_caption(frame)

        # Store the essential image file name and the new caption
        processed_data.append({
            'name': frame.get('name'),
            'raw_caption': raw_caption
        })

    return processed_data

