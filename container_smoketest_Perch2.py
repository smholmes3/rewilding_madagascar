from opensoundscape import birds_path
import bioacoustics_model_zoo as bmz

# Load the pre-trained Perch2 TensorFlow model
perch2_model = bmz.Perch2()

# Run embedding on OpenSoundscape's built-in example audio
embeddings = perch2_model.embed(birds_path)

print("Perch2 embed ran successfully.")