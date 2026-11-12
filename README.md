# Query-Guided Video Frame Retrieval

A local research prototype for query-guided video frame retrieval. It samples a selected video at one frame per second, ranks frames with SigLIP2 text–image similarity, and presents the score timeline and a paginated gallery of sampled frames. Gemini can optionally rewrite a plain-language request as visual search phrases before scoring.

The app finds moments described in the video. It does not determine whether a crime occurred.

## Run locally

1. Install Python 3.10 or newer. For GPU inference, install the PyTorch build that matches your CUDA setup from the [official PyTorch install selector](https://pytorch.org/get-started/locally/). CPU inference also works, but takes longer.
2. Create and activate a virtual environment, then install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env`. Set `VIDEO_DIR` to the folder containing the videos. Set `GEMINI_API_KEY` to a key from [Google AI Studio](https://aistudio.google.com/apikey).
4. Start the app:

   ```bash
   uvicorn app:app --host 127.0.0.1 --port 8000
   ```

5. Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

The first search downloads the SigLIP2 model if it is not already cached. Without a Gemini key, searches still work using the raw query. The key stays in the Python backend and is never sent to the browser.

## Run the app on a Colab GPU

Use `Query_Guided_Video_Retrieval_Colab_GPU.ipynb` with a Colab runtime set to GPU. Run its single setup cell, upload `query_guided_video_retrieval_colab.zip` when prompted, authorize Drive, and enter the Drive folder containing your videos. The cell keeps Colab's CUDA-enabled PyTorch installation, starts this same FastAPI app, and prints a link to its UI. The header reports the GPU name when CUDA is active. The notebook prompts for the model and an optional Gemini API key without displaying or saving the key. To use a fine-tuned `.safetensors` checkpoint, upload it to Drive first and enter its mounted Drive path when prompted.

The notebook uses the base SigLIP2 model by default unless `/content/drive/MyDrive/model.safetensors` exists or you enter a different model/checkpoint path at its prompt. A full `.safetensors` file must contain the complete model state; adapter-only checkpoints need a different loading path.

## Use a fine-tuned SigLIP2 checkpoint

For a full-model `.safetensors` file fine-tuned from `google/siglip2-base-patch16-224`, set the weights path and base architecture in `.env`:

```dotenv
SIGLIP_MODEL=/path/to/model.safetensors
SIGLIP_BASE_MODEL=google/siglip2-base-patch16-224
```

The app loads the processor and architecture from the base model, then strictly loads the checkpoint weights. A standalone weights file must contain the complete model state; adapter-only files need a different loading path. Hugging Face checkpoint directories with their own config and processor files can still be used by setting only `SIGLIP_MODEL` to that directory. Restart the backend after changing the model path.

## What the cost comparison means

The UI estimates sending one resized image per second to a VLM, compared with sending only the selected top-ranked frames. It counts 258 image tokens per frame by default, which matches Google's documented image-token count for images no larger than 384 pixels in either dimension. The estimate includes the entered prompt and answer token assumptions for each configured request batch. The default rate fields use Gemini 2.5 Flash standard paid rates as an example: $0.30 per million input tokens and $2.50 per million output tokens. Rates are editable in the UI. The model name is a label; changing it does not update the rates automatically.

This is a planning estimate, not a billing quote. Actual tokenization, frame resolution, request batching or limits, free-tier eligibility, and model pricing can change the result. It excludes Gemini query-rewrite cost and local SigLIP2 compute. The gallery can download the selected frames as a ZIP with a `manifest.csv` containing the query, timestamp, rank, and score. Check Google's [current Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing) and [token-counting guidance](https://ai.google.dev/gemini-api/docs/tokens) before using the estimate in a report.

## Components

- `app.py` — FastAPI API, Gemini query rewriting, video sampling, SigLIP2 embedding and scoring, frame previews.
- `static/index.html`, `static/styles.css`, `static/app.js` — browser interface, score chart, gallery and estimate controls.
- `requirements.txt` — Python dependencies.
