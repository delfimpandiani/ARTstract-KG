import json

# Load your JSON data
with open('merged_ARTstract.json', 'r') as json_file:
    data = json.load(json_file)


# Function to generate the basic text description
def generate_basic_description(image_data):
    art_style = image_data['as']['ARTstract_as_2023_06_26']['art_style']
    emotion = image_data['em']['ARTstract_em_2023_06_26']['emotion']
    action = image_data['act']['ARTstract_act_2023_06_28']['action_label']

    colors = [color['webcolor_name'] for color in image_data['color']['ARTstract_color_2023_06_26'][:5]]
    color_text = ', '.join(colors)

    objects = [obj['detected_object'] for obj in image_data['od']['ARTstract_od_2023_06_28']['detected_objects']]
    object_text = ', '.join(objects)

    human_presence = image_data['hp']['ARTstract_hp_2023_06_26']['human_presence']
    age_tier = image_data['age']['ARTstract_age_2023_06_26']['age_tier']
    caption = image_data['ic']['ARTstract_ic_2023_06_28']['image_description']

    description = f"This image comes from {image_data['source_dataset']}, shows a {art_style} art style, evokes {emotion} emotion, depicts {action}, the top five colors are: {color_text}, and the following objects were detected: {object_text}. It has a human presence: {human_presence}. Caption: '{caption}'. Age tier: {age_tier}."

    return description


# Loop through each image data
for image_id, image_data in data.items():
    basic_text = generate_basic_description(image_data)

    # Print or save the generated texts as needed
    print(f"Image {image_id} (Basic Description):\n{basic_text}\n")
