import argparse
import os
import requests
from dotenv import load_dotenv
from mlx_nougat.cli import generate, extract_pdf_pages_as_images
from transformers import NougatProcessor
from mlx_nougat.nougat import Nougat
import mlx.core as mx

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

load_dotenv()
API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


DEFAULT_HEADERS = {
    "x-api-key": ANTHROPIC_API_KEY,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json",
}
DEFAULT_MODEL = "claude-3-5-sonnet-20240620"


def call_agent(
    role: str,
    goal: str,
    task: str,
    model: str = DEFAULT_MODEL,
) -> str:
    data = {
        "model": model,
        "max_tokens": 4000,
        "temperature": 0,
        "system": f"You are a {role}, your goal is to {goal}.",
        "messages": [{"role": "user", "content": f"here is the task: {task}"}],
    }
    headers = DEFAULT_HEADERS.copy()
    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        completion_text = response_data["content"][0]["text"].strip()
        return completion_text
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error calling Anthropic API: {e}")
    

def review_section32(section32_text:str):
    lawyer_review = call_agent(
    role="real estate lawyer",
    goal="conduct a comprehensive legal review of the Section 32 Vendor's Statement for a property purchase, identifying any risks, liabilities or issues that could negatively impact the buyer",
    task=f"""
        Please review the following Section 32 Vendor's Statement:
        ```
        {section32_text}
        ```
        As a real estate lawyer, please analyze this document in detail and provide a thorough legal opinion for the prospective buyer. Identify and explain any legal risks, liabilities, or problematic issues disclosed in the statement that the buyer should be aware of.

        Key areas to focus on in your review:

        1. Title & Ownership: Carefully examine the title details and registered proprietor information. Note any encumbrances, caveats, or other issues that could affect clear title transfer.

        2. Easements, Covenants & Restrictions: Assess the implications of any registered easements, covenants, or restrictions on title. Explain how these may limit the buyer's use or development of the property. 

        3. Zoning & Planning Controls: Review the property's zoning and any applicable planning overlays or restrictions. Advise how current zoning and planning regulations may impact the buyer's intended use or future plans for the property.

        4. Notices & Orders: Analyze any disclosed notices, orders, or proposals from authorities (e.g. local council, government departments). Explain the nature of these notices and potential legal obligations or liabilities they may create for the new owner.

        5. Required Inspections & Reports: Evaluate the currency and contents of any required inspection reports, such as building, pest, or strata reports. Identify any major issues flagged in these reports and the potential legal implications for the buyer.

        For each issue found, provide:
        - A clear explanation of the legal issue and specific clause or section it relates to 
        - An assessment of the potential legal consequences and level of risk it poses to the buyer
        - Recommendations on how the buyer should address the issue and protect their interests

        Close with an overall opinion on the key legal risks in the statement and whether you would advise the buyer to proceed with the purchase based on your findings. Recommend any specific conditions, legal protections or further investigations the buyer should pursue before finalizing the purchase.
        """,
        model="claude-3-haiku-20240307"
    )
    buyer_agent_review = call_agent(
        role="buyer's real estate agent",
        goal="thoroughly review the Section 32 Vendor's Statement and advise the buyer on any concerning issues that could affect the property's value, use or future potential",
        task=f"""
            Please review the following Section 32 Vendor's Statement for a property your client is considering purchasing: 
            ```
            {section32_text}
            ```
            As the buyer's agent, your role is to identify any red flags or issues in this document that could negatively impact your client's interests. Examine each section closely and provide a detailed assessment.

            Specific aspects to focus on:
            
            1. Ownership & Title: Note any issues with the vendor's title or capacity to sell. Look for any encumbrances or restrictions that could affect the buyer's clear title. 

            2. Restrictions on Use or Development: Carefully assess any registered or unregistered easements, covenants and restrictions. Advise how these may limit the buyer's intended use or future development of the land.

            3. Zoning & Overlays: Analyze the property's zoning and any applicable planning overlays or controls. Explain any restrictions these place on use or development and how they align with the buyer's plans.

            4. Rates, Charges & Outgoings: Confirm all rates, taxes and charges disclosed are accurate and paid up to date. Note any unusual costs or liabilities.

            5. Notices or Orders: Examine any notices or orders disclosed from authorities. Assess the implications of these for the buyer in terms of costs, repairs or limitations on use of the property. 

            6. Required Inspections: Review the results of building, pest and strata inspection reports. Summarize any significant defects or repair issues and estimate the cost implications for the buyer.

            For each point of concern, provide:
            - A summary of the issue and where it is found in the document 
            - An explanation of why it is problematic from the buyer's perspective
            - The potential impact on the property's value, use or development potential
            - Recommendations on how the buyer should address or investigate the issue further

            Conclude with your overall advice to the buyer on the key issues found and whether you believe they should proceed with the purchase based on these vendor disclosures. Suggest any further investigations, negotiations or conditions the buyer should consider to protect their position.
        """,
        model="claude-3-haiku-20240307"
    )    

    conveyancer_review = call_agent(
        role="licensed conveyancer", 
        goal="meticulously review the Section 32 Vendor's Statement, assessing compliance and identifying any gaps, errors or legal issues the buyer must address",
        task=f"""
            Please conduct a thorough compliance review of the following Section 32 Vendor's Statement:
            ```
            {section32_text}
            ```
            Assess each component of the statement in detail to determine if it meets the legal requirements for vendor disclosure in a property sale. Identify any missing information, errors, misrepresentations or areas of non-compliance the buyer should be aware of.

            Key aspects to scrutinize:

            1. Vendor Details & Title Particulars: Check the accuracy of the vendor's title details and capacity to sell. Note any anomalies in the title description or particulars.

            2. Statutory Disclosures: Confirm all required statutory disclosures are present and compliant, including zoning, planning controls, rates & outgoings, notices & orders.
            
            3. Registered Encumbrances: Carefully examine the details of any encumbrances registered on title, like mortgages, caveats, easements and covenants. Advise if any are unusual or concerning.

            4. Unregistered Dealings: Assess the nature and implications of any unregistered dealings disclosed, like leases, licences or other agreements affecting the property.

            5. Goods & Chattels: Review the description of any movable goods and chattels included in the sale. Note any lack of clarity or potential disputes.

            6. Inspection Reports: Check if required inspection reports like building and pest certificates are complete, current and included as annexures.

            For each compliance issue found, provide:
            - The specific section and clause it relates to
            - A clear explanation of what is missing, incorrect or non-compliant 
            - The potential legal implications or risks this creates for the buyer
            - Recommendations on how the buyer can rectify the issue or protect their interests
            
            Conclude with a summary of the key areas of non-compliance or concern in the vendor statement. Advise the buyer on the overall level of legal risk these issues present and recommend any amendments, further searches or conditions that should be addressed before proceeding with the purchase.

            """,
            model="claude-3-haiku-20240307"
    )       

    refined_output = call_agent(
        role="Principal Lawyer & Conveyancing Specialist", 
        goal="Provide a comprehensive review of the Section 32 Vendor's Statement based on the legal, buyer's agent and conveyancer assessments, giving detailed advice to the buyer on all risks and recommendations",
        task=f"""
            As an expert in property law and conveyancing, your task is to review the Section 32 Vendor's Statement for a property purchase in light of the following professional assessments:

            Lawyer's Review: 
            {lawyer_review}

            Buyer's Agent Review:
            {buyer_agent_review}
            
            Conveyancer's Review:
            {conveyancer_review}

            Analyze the key issues raised and develop a comprehensive advisory report for the buyer. Your report should include:

            Executive Summary: 
            - Summarize the 3-4 most significant legal risks or concerns with the property and vendor statement
            - Provide your overall opinion on whether the buyer should proceed with the purchase based on these issues

            Detailed Findings:
            For each major issue found in the professional reviews, provide:
            1. Issue Overview: Concise explanation of the issue and its practical implications for the buyer 
            2. Legal Context: Relevant legislation, case law or legal principles that apply
            3. Risk Assessment: Your evaluation of the level of legal and financial risk the issue poses to the buyer
            4. Recommendations: Detailed advice on how the buyer should address the issue, including:
                - Further investigations or reports required
                - Searches or legal inquiries to conduct
                - Amendments or conditions to request in the contract
                - Other protective measures to mitigate risk
            5. Implications for Purchase Decision: Your professional opinion on whether this issue is:
                - Unlikely to be a barrier if recommendations are implemented
                - A significant concern that may warrant reconsideration of the purchase
                - A 'deal-breaker' that should make the buyer terminate the purchase
            
            Conclusions & Next Steps:
            - Reiterate the key legal risks and implications of proceeding with the purchase
            - Recommend specific next steps the buyer should take in order of priority
            - Advise on any critical timelines or deadlines for action to protect the buyer's position

            The buyer is relying on your expertise to identify and explain the salient legal issues with this property purchase and develop a clear strategy to address them. Provide as much practical, actionable advice as possible to help them make a fully informed decision and navigate the conveyancing process from a position of knowledge and legal protection.
            """
    )  
    return refined_output


def main():
    parser = argparse.ArgumentParser(description="Process a Section 32 document from a PDF file, extract text using the Nougat model, and generate a comprehensive review using AI agents.")
    parser.add_argument("--model", default="mzbac/nougat-base-8bit-mlx", help="Model name or path")
    parser.add_argument("--input", required=True, help="Input PDF file path")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top p for generation")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Repetition penalty for generation")
    parser.add_argument("--output", help="Output file path for saving reviewed result")

    args = parser.parse_args()

    nougat_processor = NougatProcessor.from_pretrained(args.model)
    model = Nougat.from_pretrained(args.model)

    images = extract_pdf_pages_as_images(args.input)
    results = []
    for i, img in enumerate(images):
        print(f"Processing page {i+1}/{len(images)}")
        pixel_values = mx.array(nougat_processor(img, return_tensors="np").pixel_values).transpose(0, 2, 3, 1)
        outputs = generate(model, pixel_values, max_new_tokens=4096, 
                           eos_token_id=nougat_processor.tokenizer.eos_token_id,
                           temperature=args.temperature,
                           top_p=args.top_p,
                           repetition_penalty=args.repetition_penalty)
        results.append(nougat_processor.tokenizer.decode(outputs))
    
    section32_text = "\n\n".join(results)
    reviewed_text = review_section32(section32_text)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(reviewed_text)
        print(f"Reviewed result saved to {args.output}")
    else:
        print(reviewed_text)

if __name__ == "__main__":
    main()
