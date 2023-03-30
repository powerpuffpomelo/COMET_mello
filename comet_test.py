from comet import download_model, load_from_checkpoint
import time

# model_path = download_model("wmt20-comet-da")   # /home/tiger/.cache/torch/unbabel_comet/wmt20-comet-da/checkpoints/model.ckpt
model_path = '/home/tiger/.cache/torch/unbabel_comet/wmt20-comet-da/checkpoints/model.ckpt'
model = load_from_checkpoint(model_path)
data = [
    {
        "src": "Dem Feuer konnte Einhalt geboten werden",
        "mt": "The fire could be stopped",
        "ref": "They were able to control the fire."
    },
    {
        "src": "Schulen und Kindergärten wurden eröffnet.",
        "mt": "Schools and kindergartens were open",
        "ref": "Schools and kindergartens opened"
    },
    {
        "src": "Dem Feuer konnte Einhalt geboten werden",
        "mt": "The fire could be stopped",
        "ref": "They were able to control the fire."
    },
    {
        "src": "Schulen und Kindergärten wurden eröffnet.",
        "mt": "Schools and kindergartens were open",
        "ref": "Schools and kindergartens opened"
    },
    {
        "src": "Dem Feuer konnte Einhalt geboten werden",
        "mt": "The fire could be stopped",
        "ref": "They were able to control the fire."
    },
    {
        "src": "Schulen und Kindergärten wurden eröffnet.",
        "mt": "1",
        "ref": "Schools and kindergartens opened"
    },
    {
        "src": "Dem Feuer konnte Einhalt geboten werden",
        "mt": "The fire could be stopped",
        "ref": "They were able to control the fire."
    },
    {
        "src": "Schulen und Kindergärten wurden eröffnet.",
        "mt": "Schools and kindergartens were open",
        "ref": "Schools and kindergartens opened"
    },
    {
        "src": "Dem Feuer konnte Einhalt geboten werden",
        "mt": "1",
        "ref": "They were able to control the fire."
    },
    {
        "src": "Schulen und Kindergärten wurden eröffnet.",
        "mt": "1",
        "ref": "Schools and kindergartens opened"
    },
    {
        "src": "Dem Feuer konnte Einhalt geboten werden",
        "mt": "The fire could be stopped",
        "ref": "They were able to control the fire."
    },
    {
        "src": "Schulen und Kindergärten wurden eröffnet.",
        "mt": "Schools and kindergartens were open",
        "ref": "Schools and kindergartens opened"
    },
]

t1 = time.time()

seg_scores, sys_scores = model.predict(data, batch_size=8)
t2 = time.time()
print("=================================================================")
print(seg_scores)
print(t2 - t1)

seg_scores = model.predict_mello(data, batch_size=8, device=1)
t3 = time.time()
print(seg_scores)
print(t3 - t2)


# python3 comet_test.py