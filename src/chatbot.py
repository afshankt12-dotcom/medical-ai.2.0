import torch

from huggingface_hub import hf_hub_download

from src.retrieval import Retriever
from src.tokenizer import MedicalTokenizer
from src.transformer_model import MedicalTransformer


# Hugging Face Model Repository
HF_REPO = "shanufewf/medical-transformer-model"


# Download model files automatically
MODEL_PATH = hf_hub_download(
    repo_id=HF_REPO,
    filename="medical_transformer.pth"
)

VOCAB_PATH = hf_hub_download(
    repo_id=HF_REPO,
    filename="vocab.json"
)


class MedicalChatbot:

    def __init__(self):

        print("Loading Medical Chatbot...")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )


        # -----------------------------
        # Load Tokenizer
        # -----------------------------

        self.tokenizer = MedicalTokenizer()

        self.tokenizer.load_vocab(
            VOCAB_PATH
        )


        # -----------------------------
        # Load Transformer Model
        # -----------------------------

        self.model = MedicalTransformer(
            vocab_size=self.tokenizer.vocab_size()
        )


        self.model.load_state_dict(
            torch.load(
                MODEL_PATH,
                map_location=self.device
            )
        )


        self.model.to(
            self.device
        )

        self.model.eval()


        # -----------------------------
        # Retrieval System
        # -----------------------------

        self.retriever = Retriever()


        print("Medical Chatbot Ready!")


    # -----------------------------------------
    # Chatbot Answer
    # -----------------------------------------

    def ask(self, question):

        answer, score = self.retriever.search(
            question
        )


        return {

            "question": question,

            "answer": answer,

            "confidence": round(
                score,
                3
            )

        }