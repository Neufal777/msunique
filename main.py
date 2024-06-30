import os
import fitz
import openai
import json

# OpenAI configuration
openai.api_type = os.getenv("OPENAI_API_TYPE", "default_api_type")
openai.api_base = os.getenv("OPENAI_API_BASE", "default_api_base")
openai.api_key = os.getenv("OPENAI_API_KEY", "default_api_key")
openai.api_version = os.getenv("OPENAI_API_VERSION", "default_api_version")


# Function to read PDF and generate prompts for each segment of 48 pages
def process_pdf_and_generate_prompts(pdf_path, segment_size=48):
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    segments = range(0, total_pages, segment_size)

    all_responses = []

    for i, start_page in enumerate(segments):
        end_page = min(start_page + segment_size, total_pages)
        segment_text = ""
        for page_num in range(start_page, end_page):
            page = doc.load_page(page_num)
            segment_text += page.get_text()

        # Generate prompt for the current segment
        prompt = (
                "Based on the information extracted from the PDF segment, "
                "please provide a detailed response."
                "\n\nPDF Segment Text:\n\n"
                + segment_text
                + "\n\nResponse:"
        )

        response = openai.ChatCompletion.create(
            engine="gpt-4o-East-US2",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        # Print or process the response
        print(f"Prompt generated for pages {start_page + 1}-{end_page}:")
        print(response['choices'][0]['message']['content'])
        print("-" * 50)

        all_responses.append({
            f"segment_{i + 1}": {
                "response": response['choices'][0]['message']['content']
            }
        })

    doc.close()

    # After processing segments, generate a "general" prompt with all responses combined
    general_prompt = (
        "Based on the data extracted from all segments of the PDF, "
        "please provide a comprehensive summary or additional insights."
        "\n\nin a json format, dont respond with anything else, no commas, no text, just the pure json.:\n\n"
    )

    # Append all segment responses to the general prompt
    for idx, response in enumerate(all_responses, start=1):
        general_prompt += f"Segment {idx}:\n"
        general_prompt += response[f"segment_{idx}"]['response'] + "\n\n"

    # Generate "general" prompt response
    general_response = openai.ChatCompletion.create(
        engine="gpt-4o-East-US2",  # Specify your deployment name
        messages=[
            {"role": "user", "content": general_prompt}
        ]
    )

    # Print or process the general response
    print("General prompt response:")
    print(general_response['choices'][0]['message']['content'])
    print("-" * 50)

    # Prepare the final JSON object to save to file
    final_json = {
        "General_Prompt_Response": json.loads(general_response['choices'][0]['message']['content'].replace('`',''))
    }

    # Write the final JSON object to a JSON file named final_analysis.json
    output_file = "final_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, ensure_ascii=False, indent=4)

    print(f"General prompt response saved to {output_file}")


# Example usage
if __name__ == "__main__":
    pdf_file = "Data/IBM/2023.pdf"  # Replace with your PDF file path
    process_pdf_and_generate_prompts(pdf_file)
