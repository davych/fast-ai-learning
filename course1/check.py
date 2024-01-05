from fastai.learner import load_learner
from fastai.vision.core import PILImage

learn = load_learner('course1/export.pkl')

im = PILImage.create('course1/forest.jpg')
print(im)
is_bird,_,probs = learn.predict(im)
print(f"This is a: {is_bird}.")
print(f"Probability it's a bird: {probs[0]:.4f}")