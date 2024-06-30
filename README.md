# Section 32 Document Analyzer

This project provides a comprehensive tool for analyzing Section 32 documents from PDF files. It uses the Nougat model to extract text from PDFs and leverages AI agents to generate a detailed review of the document's contents.

## Features

- Extract text from PDF files using the Nougat model
- Generate comprehensive reviews of Section 32 documents using AI agents
- Analyze the document from multiple perspectives: legal, buyer's agent, and conveyancer
- Provide a refined output highlighting potential issues and areas of concern

## Requirements

- Python 3.7+
- MLX Nougat
- Transformers
- Requests
- python-dotenv

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install mlx-nougat transformers requests python-dotenv
```

3. Set up your Anthropic API key in a `.env` file:

```bash
ANTHROPIC_API_KEY=your_api_key_here
```

## Usage

Run the script from the command line with the following options:

```bash
python main.py --input path/to/your/section32/pdf --output path/to/output/file.txt
```

Optional arguments:

- `--model`: Specify the Nougat model to use (default: "mzbac/nougat-base-8bit-mlx")
- `--temperature`: Set the temperature for text generation (default: 0.3)
- `--top_p`: Set the top p value for text generation (default: 0.95)
- `--repetition_penalty`: Set the repetition penalty for text generation (default: 1.2)

## How it works

1. The script extracts text from the provided PDF using the Nougat model.
2. It then uses AI agents to analyze the extracted text from different perspectives:
   - Legal review
   - Buyer's agent review
   - Conveyancer review
3. The results are refined and combined into a comprehensive report highlighting potential issues and areas of concern.
4. The final report is either printed to the console or saved to the specified output file.

## Limitations

The results generated by this tool can be inaccurate and highly depend on the quality of the PDF-to-text extraction process. Current extraction methods don't work well with diagrams, which may lead to incomplete or misinterpreted information. This tool should be used as a supplementary aid only and not as a substitute for professional legal advice. Always consult with qualified professionals when making important decisions related to property transactions.