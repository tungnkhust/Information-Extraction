from src.coref.NeuralCoreF import NeuralCoreF


model = NeuralCoreF()
output = model.run("My name is Tung , I study at Standford")
print(output)