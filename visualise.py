import torch
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from model import Encoder, Decoder, Seq2Seq
from dataset import CharVocab
from config import get_args

args  = get_args()

app = FastAPI()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load vocab and model
input_vocab = CharVocab.load("saved/input_vocab.pt")
output_vocab = CharVocab.load("saved/output_vocab.pt")

encoder = Encoder(len(input_vocab), 
                    embed_dim=args.input_embedding,
                    hidden_dim=args.hidden_layer_size,
                    num_layers=args.encoder_layers, 
                    rnn_type=args.cell_type,
                    bidirectional=args.encoder_bidirectional, 
                    dropout=args.dropout)
decoder = Decoder(len(output_vocab), 
                      embed_dim=args.input_embedding, 
                      hidden_dim=args.hidden_layer_size, 
                      num_layers=args.decoder_layers, 
                      rnn_type=args.cell_type,
                      bidirectional=args.decoder_bidirectional,
                      dropout=args.dropout,
                      use_attention=args.use_attention,
                      encoder_output_dim=args.hidden_layer_size * 2 if args.encoder_bidirectional else args.hidden_layer_size)
model = Seq2Seq(encoder, decoder, rnn_type=args.cell_type).to(DEVICE)
model.load_state_dict(torch.load("saved/seq2seq_model_attention.pt", map_location=DEVICE))
model.eval()


@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Latin to Devanagari</title>
        <link href="https://fonts.googleapis.com/css2?family=Roboto+Mono&display=swap" rel="stylesheet">
        <style>
            body {
                font-family: 'Roboto Mono', monospace;
                background-color: #f9f9f9;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
            }
            .container {
                background-color: #ffffff;
                padding: 40px;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                text-align: center;
                max-width: 500px;
                width: 100%;
            }
            h2 {
                margin-bottom: 20px;
                color: #333;
            }
            input[type="text"] {
                width: 80%;
                padding: 10px;
                font-size: 16px;
                border: 1px solid #ccc;
                border-radius: 6px;
                margin-bottom: 20px;
                font-family: inherit;
            }
            button {
                padding: 10px 20px;
                font-size: 16px;
                background-color: #00bfa5;
                color: white;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                transition: background-color 0.2s ease-in-out;
            }
            button:hover {
                background-color: #009e88;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Latin → Devanagari Translator</h2>
            <form action="/translate" method="get">
                <input type="text" name="word" placeholder="Enter Latin word" required>
                <br>
                <button type="submit">Translate</button>
            </form>
        </div>
    </body>
    </html>
    """


@app.get("/translate", response_class=HTMLResponse)
async def translate(word: str):
    input_tensor = input_vocab.tensor_from_text(word).to(DEVICE)

    with torch.no_grad():
        output, attention_weights = model.translate(input_tensor, output_vocab, max_len=30)
        output_word = output_vocab.decode(output)

        decoded_input = [input_vocab.idx2char[idx] for idx in input_tensor.squeeze(0).cpu().numpy()]
        decoded_output = [output_vocab.idx2char[idx] for idx in output]

        decoded_input = [c for c in decoded_input if c not in ['<sos>', '<eos>']]
        decoded_output = [c for c in decoded_output if c not in ['<sos>', '<eos>']]


        attention_weights = np.array(attention_weights).squeeze(1)[:,2:]
        js_attention = [
            [round(w, 4) for w in attention_weights[i].tolist()]
            for i in range(len(decoded_output))
        ]

        return generate_interactive_attention_html(decoded_input, decoded_output, js_attention, word, output_word)


def generate_interactive_attention_html(input_sentence, output_sentence, attention, latin_word, devanagari_word):
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Attention Visualizer</title>
        <style>
            body {{
                font-family: 'Roboto Mono', monospace;
                background-color: #f9f9f9;
                display: flex;
                justify-content: center;
                padding: 40px;
            }}
            .container {{
                background-color: white;
                padding: 30px 40px;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                max-width: 700px;
                width: 100%;
                text-align: center;
            }}
            h2 {{
                font-weight: 700;
                margin-bottom: 25px;
            }}
            .centered-text {{
                font-size: 20px;
                font-weight: 500;
                margin: 5px 0 15px;
            }}
            .devanagari {{
                font-size: 24px;
                color: #333;
            }}
            .output-char {{
                display: inline-block;
                margin: 5px;
                padding: 8px 14px;
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 8px;
                cursor: pointer;
                font-size: 18px;
                transition: background-color 0.2s ease;
            }}
            .output-char:hover {{
                background-color: #d0eaff;
            }}
            #input-highlight {{
                font-family: monospace;
                font-size: 18px;
                margin-top: 20px;
                padding-top: 15px;
                border-top: 1px solid #eee;
            }}
            .char-box {{
                display: inline-block;
                padding: 4px 6px;
                margin: 1px;
                border-radius: 4px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Latin → Devanagari Attention Visualizer</h2>

            <p><strong>Input:</strong></p>
            <p class="centered-text">{latin_word}</p>

            <p><strong>Output:</strong></p>
            <p class="centered-text devanagari">{devanagari_word}</p>

            <div><strong>Output characters (hover to see attention):</strong></div>
            <div id="output-chars">
    """

    for i, c in enumerate(output_sentence):
        html += f'<span class="output-char" onmouseover="showAttention({i})">{c}</span>'

    html += f"""
            </div>
            <div id="input-highlight"></div>
        </div>

        <script>
            const inputChars = {input_sentence};
            const attention = {attention};

            function showAttention(index) {{
                const weights = attention[index];
                let html = "";
                for (let i = 0; i < inputChars.length; i++) {{
                    const w = weights[i];
                    const bg = `rgba(124, 252, 0, ${{w}})`;
                    html += `<span class="char-box" style="background-color:${{bg}}">${{inputChars[i]}}</span>`;
                }}
                document.getElementById("input-highlight").innerHTML = html;
            }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)