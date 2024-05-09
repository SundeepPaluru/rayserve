from starlette.requests import Request

from ray import serve
from sentence_transformers import SentenceTransformer


@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0.5, "num_gpus": 0})
class SentenceTransformerModel:
    def __init__(self):
        self.model = SentenceTransformer('intfloat/e5-base-v2')
        # self.model.cuda()

    async def __call__(self, request: Request):
        sentences = await request.json()
        return self.model.encode(sentences["text"])


sentence_embedding_app = SentenceTransformerModel.bind()
