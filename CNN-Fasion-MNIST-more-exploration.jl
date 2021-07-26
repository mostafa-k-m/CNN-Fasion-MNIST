### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ 99ee4fa6-a080-11eb-1df7-1b2a6abf52a7
begin 
	using Flux, PlutoUI, Statistics, MLDatasets, Images
	using Flux.Data: DataLoader 
	using Flux: @epochs, onehotbatch, onecold, logitcrossentropy, throttle, unsqueeze
	using Random: shuffle, randperm
	using StatsBase: countmap, proportionmap
	using IterTools:ncycle
	using CUDA
end

# ╔═╡ 55f2750a-bbf8-448f-b900-57b37e235ba5
using Plots

# ╔═╡ 9ddb04c6-6831-458e-a1ea-da0e5a079d9a
md"# Bonus ConvNet MNIST training exploration"

# ╔═╡ 2f8758e0-c8eb-4b83-9f6b-b8e707651a6e
md"## Loading the Dataset"

# ╔═╡ 16b69eab-9e61-4941-82e7-d69fc69a7221
CUDA.reclaim()

# ╔═╡ 46437f73-2f7a-4c39-ae62-0bd199e652b5
begin
	train_x_org_, train_y_i = MNIST.traindata(Float32);
	test_x_org_, test_y_i = MNIST.testdata(Float32);
end

# ╔═╡ 70283569-b190-4c1b-9379-7221c4469b3a
begin
	train_y, test_y = onehotbatch(train_y_i, 0:9)|> gpu, onehotbatch(test_y_i, 0:9)|> gpu
	train_x_org =  reshape(train_x_org_, 28, 28, 1, :) |> gpu
	test_x_org =  reshape(test_x_org_, 28, 28, 1, :) |> gpu
end

# ╔═╡ d1ea9c1f-d13c-4487-aa3b-eba6838b596c
md"# Question 1"

# ╔═╡ c884fbfd-32f3-4ba1-9cad-a30626b2032b
md"## Defining the Model"

# ╔═╡ ab7aa26a-b8e7-435f-ae12-27e5629aac11
model = Chain(
    Conv((3, 3), 1 => 32, pad=(1, 1), relu),
    Conv((3, 3), 32 => 64, pad=(1, 1), relu),
    MaxPool((2, 2)),
    Dropout(0.25),
    flatten,
    Dense(12544,  128, relu),
    Dropout(0.5),
	Dense(128, 10),
	softmax
) |> gpu

# ╔═╡ 72759364-0031-447c-ae2f-53bbdf952b21
md"## Defining the loss and accuracy functions"

# ╔═╡ e0bfe083-b0a7-4cd2-bebb-18140f1a786e
function loss(x, y)
	y_predict = model(x)
	return Flux.Losses.crossentropy(y_predict, y)
end

# ╔═╡ 626894c2-0dc5-4e14-860e-eb960449e6fb
function accuracy(x, y) 
	x = Flux.onecold(model(x))
	y = Flux.onecold(y)
	return mean(x .== y)
end

# ╔═╡ dbc1f2b4-270a-4de8-9df6-d914ac1630e3
function callBackEvaluation()
	lossVal = loss(test_x_org, test_y)
	accuracyVal = accuracy(test_x_org, test_y)
	@show(lossVal, accuracyVal)
end

# ╔═╡ c174095e-5fa0-47c1-8dba-fc473d69fd6d
md"## Defining the Optimizer"

# ╔═╡ 8e0e3de6-0a2e-4b92-ba5b-472a6d55141e
begin
	parameters = Flux.params(model)
	opt = Flux.Optimise.ADADelta()
end

# ╔═╡ b6ecaa59-b370-4b24-89c5-b8d7558aeb1f
md"## Loading Data and Training the Model"

# ╔═╡ d993416e-864d-4282-a59a-32a200b7199f
begin
	train_loader = Flux.Data.DataLoader((train_x_org, train_y), batchsize=50, shuffle=true)
	test_loader = Flux.Data.DataLoader((test_x_org, test_y), batchsize=50, shuffle=true)
end

# ╔═╡ 54aa1adc-32b3-4d23-8c98-53e6242b15ac
Flux.Optimise.train!(
	loss,
	parameters,
	ncycle(train_loader, 10),
	opt,
	cb=Flux.throttle(callBackEvaluation, 10)
)

# ╔═╡ fe4370e1-c66d-4dbc-95f2-34b9ca64eef5
md"## Model Accuracy"

# ╔═╡ 84dbd34a-5d7f-4351-8a74-92cceb4977f0
accuracy(test_x_org, test_y)

# ╔═╡ 9f0a989c-2e8d-4a34-8694-243761bcd947
md"## Comment:
The accuracy of our model in julia is very close to the provided tensorflow example. the difference of less than .3% in accuracy is negligible."

# ╔═╡ 75eac38b-367b-4f5b-91b1-d36dfa08f63c
md"# Question 2"

# ╔═╡ 29c15789-dadb-4601-9bda-ae6ac67d4567
md"## Shuffling the each image pixel
it is worth noting that all images are being shuffled the same exact way. 
The pix_perm variable is randomized but once it is compiled it is essentially a constant map of where pixels should be that when applied to an image it would do the same kind of computation."

# ╔═╡ 988eb0aa-2c20-4363-9e80-426a79d3a264
pix_perm=reshape(CartesianIndices((28,28))[randperm(28*28)], 28,28);

# ╔═╡ ed134f6f-ad62-4bbb-87b3-55a6e10e52ef
begin
	train_x_ = similar(train_x_org_);
	for i=1:size(train_x_,3)
		train_x_[:,:,i] .= train_x_org_[:,:,i][pix_perm]
	end
	train_x = reshape(train_x_, 28, 28, 1, :) |> gpu
end

# ╔═╡ 325247ee-ce20-403b-a340-4f2d5a10763a
begin
	test_x_ = similar(test_x_org_);
	for i=1:size(test_x_,3)
	    test_x_[:,:,i] .= test_x_org_[:,:,i][pix_perm]
	end
	test_x = reshape(test_x_, 28, 28, 1, :) |> gpu
end

# ╔═╡ fd25f18d-33fa-4ed4-8d27-2d937f50211f
md"## Defining a New model to be trained with the shuffled images"

# ╔═╡ 9aad5f63-012a-44f9-b8e9-6d784446aabc
model_shuf = Chain(
    Conv((3, 3), 1 => 32, pad=(1, 1), relu),
    Conv((3, 3), 32 => 64, pad=(1, 1), relu),
    MaxPool((2, 2)),
    Dropout(0.25),
    flatten,
    Dense(12544,  128, relu),
    Dropout(0.5),
	Dense(128, 10),
	softmax
) |> gpu

# ╔═╡ d89a57cf-9848-4089-b7d4-06966c67d6bc
function accuracyShuf(x, y) 
	x = Flux.onecold(model_shuf(x))
	y = Flux.onecold(y)
	return mean(x .== y)
end

# ╔═╡ 18ed7a99-017f-4cd7-8154-5054175c2288
function lossShuf(x, y)
	y_predict = model_shuf(x)
	return Flux.Losses.crossentropy(y_predict, y)
end

# ╔═╡ 733b4813-98d3-40fa-8de1-172eb7d73720
function callBackEvaluationShuf()
	lossVal = lossShuf(test_x, test_y)
	accuracyVal = accuracyShuf(test_x, test_y)
	@show(lossVal, accuracyVal)
end

# ╔═╡ 7993b680-7db3-4d6c-8817-876d8913c875
begin
	parameters_shuf = Flux.params(model_shuf)
	opt_shuf = Flux.Optimise.ADADelta()
end

# ╔═╡ fa905413-fdef-4997-b916-1390e54978a2
begin
	train_loader_shuf = Flux.Data.DataLoader((train_x, train_y), batchsize=50, shuffle=true)
	test_loader_shuf = Flux.Data.DataLoader((test_x, test_y), batchsize=50, shuffle=true)
end

# ╔═╡ b1925e7e-bf68-4890-ba3d-abf9aefe5ae4
Flux.Optimise.train!(
	lossShuf,
	parameters_shuf,
	ncycle(train_loader_shuf, 10),
	opt_shuf,
	cb=Flux.throttle(callBackEvaluationShuf, 10)
)

# ╔═╡ dd1dd6ea-5539-4466-858f-b849f1b04a48
md"## Model Accuracy"

# ╔═╡ 4d65a6b4-20d5-4f37-b2ab-9b694e5519c6
accuracyShuf(test_x, test_y)

# ╔═╡ dee6784b-e2d9-4b1a-b618-1050a7aae01c
md"## Observation and Comments
In this experiment, initialization is very important. when pix_perm compiles it sets a random transofrmation that will be applied to the entire dataset. and it stands to reason that sometimes this transformation is going to be more forgiving than otherwise.

In all of our experiments we have had accuracies in the high ninties. the model trained on the shuffled images was only marginally less accurate than the model trained on the original images.

We know that CNN architecture are sensitice to variability and diversity of pixel intensity within and between local regions. It is a `spacially aware` model. in this case we did shuffle our images but we shuffled them all the same way, still giving the model a stable `pattern` to learn for each class.

Had we shuffled each individual picture randomly (compiled pix_perm in a for loop before every shuffle for example) the accuracy would have suffered greatly."

# ╔═╡ 3b551ed1-b3ba-4516-b5e6-1f6c5f53f0a7
md"# Question 3"

# ╔═╡ ecdd9297-e505-4189-8cef-c5a8bafc8901
md"## Defining a model with no dropout"

# ╔═╡ a1a3957a-8133-4930-a99f-461ff37c47f4
begin
	model_no_dropout = Chain(
	    Conv((3, 3), 1 => 32, pad=(1, 1), relu),
	    Conv((3, 3), 32 => 64, pad=(1, 1), relu),
	    MaxPool((2, 2)),
	    flatten,
	    Dense(12544,  128, relu),
		Dense(128, 10),
		softmax
	) |> gpu
	
	function accuracy_no_dropout(x, y) 
		x = Flux.onecold(model_no_dropout(x))
		y = Flux.onecold(y)
		return mean(x .== y)
	end
	
	function loss_no_dropout(x, y)
		y_predict = model_no_dropout(x)
		return Flux.Losses.crossentropy(y_predict, y)
	end
	
	parameters_no_dropout = Flux.params(model_no_dropout)
	opt_no_dropout = Flux.Optimise.ADADelta()
end

# ╔═╡ 87853196-5cda-4802-8d02-0666eed526cf
md"## This is the modek that we made in question 1 (optimal dropout)"

# ╔═╡ 9b8f60dd-890e-47ed-b1e3-bd812a2aa0b8
begin
	model_dropout = Chain(
		Conv((3, 3), 1 => 32, pad=(1, 1), relu),
		Conv((3, 3), 32 => 64, pad=(1, 1), relu),
		MaxPool((2, 2)),
		Dropout(0.25),
		flatten,
		Dense(12544,  128, relu),
		Dropout(0.5),
		Dense(128, 10),
		softmax
	) |> gpu
	
	function accuracy_dropout(x, y) 
		x = Flux.onecold(model_dropout(x))
		y = Flux.onecold(y)
		return mean(x .== y)
	end
	
	function loss_dropout(x, y)
		y_predict = model_dropout(x)
		return Flux.Losses.crossentropy(y_predict, y)
	end
	
	parameters_dropout = Flux.params(model_dropout)
	opt_dropout = Flux.Optimise.ADADelta()
end

# ╔═╡ 3322ab74-2e74-43da-b56b-dc9050911b0c
md"## Defining a model with high dropout"

# ╔═╡ ca5fdc97-7533-4697-addc-02798bc0e5ae
begin
	model_high_dropout = Chain(
		Conv((3, 3), 1 => 32, pad=(1, 1), relu),
		Conv((3, 3), 32 => 64, pad=(1, 1), relu),
		MaxPool((2, 2)),
		Dropout(0.6),
		flatten,
		Dense(12544,  128, relu),
		Dropout(0.8),
		Dense(128, 10),
		softmax
	) |> gpu
	
	function accuracy_high_dropout(x, y) 
		x = Flux.onecold(model_dropout(x))
		y = Flux.onecold(y)
		return mean(x .== y)
	end
	
	function loss_high_dropout(x, y)
		y_predict = model_dropout(x)
		return Flux.Losses.crossentropy(y_predict, y)
	end
	
	parameters_high_dropout = Flux.params(model_dropout)
	opt_high_dropout = Flux.Optimise.ADADelta()
end

# ╔═╡ 4e4da02b-2890-4bd4-a562-a528a565eebe
function customTrainingLoop(model_cust, trainLoader, test_x, test_y)
	function loss_cust(x, y)
		y_predict = model_cust(x)
		return Flux.Losses.crossentropy(y_predict, y)
	end
	
	function accuracy_cust(x, y) 
		x = Flux.onecold(model_cust(x))
		y = Flux.onecold(y)
		return mean(x .== y)
	end

	opt_cust = Flux.Optimise.ADAM()
	params_cust = Flux.params(model_cust)
	losses = []
	accuracies = []
	for (train_point, test_point) in trainLoader
		grads = gradient(params_cust) do
			loss_cust(train_point, test_point)
		end 
		Flux.Optimise.update!(opt_cust, params_cust, grads)
		append!(losses, loss_cust(test_x, test_y))
		append!(accuracies, accuracy_cust(test_x, test_y))
		@show accuracies
		if accuracies[end] > 0.95
			break
		end
	end
	return  losses, accuracies
end


# ╔═╡ cac8afd1-23c9-41eb-a9cf-aac803fd5289
md"## We will train the hhree models 
Each model will be trained for a single epoch and record the loss and the accuracy for each model"

# ╔═╡ 14bd5dc0-e515-40d3-aee2-8348a9c74100
dropout_losses, dropout_accuracies = customTrainingLoop(model_dropout, train_loader, test_x_org, test_y)

# ╔═╡ d8487d5e-8bab-40f2-a8df-4d27be2fbb77
high_dropout_losses, high_dropout_accuracies = customTrainingLoop(model_high_dropout, train_loader, test_x_org, test_y)

# ╔═╡ 32eeb4e9-f7d1-4aac-b11b-c866a4f70bf7
no_dropout_losses, no_dropout_accuracies = customTrainingLoop(model_no_dropout, train_loader, test_x_org, test_y)

# ╔═╡ 70e2ad4a-b642-4fc8-a45f-df29a3231167
function list_comp(list_)
	return [x for x in 1:length(list_)]
end

# ╔═╡ 229ebc35-f22f-4aaa-8ca6-4781895f75b8
begin
	plot(list_comp(dropout_losses), dropout_losses, label="Optimal Dropout", title="Losses")
	plot!(list_comp(high_dropout_losses), high_dropout_losses, label="High Dropout")
	plot!(list_comp(no_dropout_losses), no_dropout_losses, label="No Dropout")
end

# ╔═╡ c370e0a7-3bdf-4f98-909a-76784199fbbe
begin
	plot(list_comp(dropout_accuracies), dropout_accuracies, label="Optimal Dropout", title="Accuracies")
	plot!(list_comp(high_dropout_accuracies), high_dropout_accuracies, label="High Dropout")
	plot!(list_comp(no_dropout_accuracies), no_dropout_accuracies, label="No Dropout")
end

# ╔═╡ cbcec3a9-fdd6-465d-8d85-207feb6b78be
md"## Comments
To truly measure whether our model is over fitting, we would have to withhold a portion of the dataset for validation, some data that the model never sees during training to test the fully trained mode.
An Overfit model will perform poorly on the validation data despite having a very high training accuracy.
We couldn't acheive this in this experiment due to constraints related to my machine's gpu memory and the fact that Pluto seems to run the whole sheet and overload the memory everytime a change is made.

Therefore we will try to estimate the model's tendency to overfit by how fast the model training accuracy improves. we will use early stoping to compare  the losses and accuracies of the three models.

We notice that the model with no dropout at all was the fastest to reach 95% accuracy after just about 120 batches (of 50) of the pictures. This model is likely to overfit during the rest of the training period because of how often it will get exposed to similar data. tje model might suffer from `over-reliance` on a few of its inputs.

We can trust the dropout models to generalize better because they git rid of this `over-reliance`. Since not all of the models inputs are present at all time during training, the model does not become over-reliant on its inputs and as a result can generalize better."

# ╔═╡ df26309b-bb5a-4af8-98b6-fa1bf3accd8d
md"""
## Assignment Questions 

!!! question "Question 1 (10+ pts)"

    Replicate the Conv net described in [here](https://github.com/sambit9238/Deep-Learning/blob/master/cnn_mnist.ipynb) at `In [20]` in Julia. Do you get similar results? Comment!


"""

# ╔═╡ 0e56e1cd-4fe6-49fd-8a75-8cf1af8cb7f5
md"""

!!! question "Question 2 (5+ pts)"

    After shuffeling the pixels around, did the results change? Comment. 

!!! question "Question 2 (5+ pts)"

    Would the Conv net *overfits* faster or slower without the dropout layers? Demonstrate with results and comment. Get even more points with experimention and analysis of differet dropout rates. 



"""

# ╔═╡ Cell order:
# ╟─9ddb04c6-6831-458e-a1ea-da0e5a079d9a
# ╠═99ee4fa6-a080-11eb-1df7-1b2a6abf52a7
# ╟─2f8758e0-c8eb-4b83-9f6b-b8e707651a6e
# ╠═16b69eab-9e61-4941-82e7-d69fc69a7221
# ╠═46437f73-2f7a-4c39-ae62-0bd199e652b5
# ╠═70283569-b190-4c1b-9379-7221c4469b3a
# ╟─d1ea9c1f-d13c-4487-aa3b-eba6838b596c
# ╟─c884fbfd-32f3-4ba1-9cad-a30626b2032b
# ╠═ab7aa26a-b8e7-435f-ae12-27e5629aac11
# ╟─72759364-0031-447c-ae2f-53bbdf952b21
# ╠═e0bfe083-b0a7-4cd2-bebb-18140f1a786e
# ╠═626894c2-0dc5-4e14-860e-eb960449e6fb
# ╠═dbc1f2b4-270a-4de8-9df6-d914ac1630e3
# ╟─c174095e-5fa0-47c1-8dba-fc473d69fd6d
# ╠═8e0e3de6-0a2e-4b92-ba5b-472a6d55141e
# ╟─b6ecaa59-b370-4b24-89c5-b8d7558aeb1f
# ╠═d993416e-864d-4282-a59a-32a200b7199f
# ╠═54aa1adc-32b3-4d23-8c98-53e6242b15ac
# ╟─fe4370e1-c66d-4dbc-95f2-34b9ca64eef5
# ╠═84dbd34a-5d7f-4351-8a74-92cceb4977f0
# ╟─9f0a989c-2e8d-4a34-8694-243761bcd947
# ╟─75eac38b-367b-4f5b-91b1-d36dfa08f63c
# ╟─29c15789-dadb-4601-9bda-ae6ac67d4567
# ╠═988eb0aa-2c20-4363-9e80-426a79d3a264
# ╠═ed134f6f-ad62-4bbb-87b3-55a6e10e52ef
# ╠═325247ee-ce20-403b-a340-4f2d5a10763a
# ╟─fd25f18d-33fa-4ed4-8d27-2d937f50211f
# ╠═9aad5f63-012a-44f9-b8e9-6d784446aabc
# ╠═d89a57cf-9848-4089-b7d4-06966c67d6bc
# ╠═18ed7a99-017f-4cd7-8154-5054175c2288
# ╠═733b4813-98d3-40fa-8de1-172eb7d73720
# ╠═7993b680-7db3-4d6c-8817-876d8913c875
# ╠═fa905413-fdef-4997-b916-1390e54978a2
# ╠═b1925e7e-bf68-4890-ba3d-abf9aefe5ae4
# ╟─dd1dd6ea-5539-4466-858f-b849f1b04a48
# ╠═4d65a6b4-20d5-4f37-b2ab-9b694e5519c6
# ╟─dee6784b-e2d9-4b1a-b618-1050a7aae01c
# ╟─3b551ed1-b3ba-4516-b5e6-1f6c5f53f0a7
# ╟─ecdd9297-e505-4189-8cef-c5a8bafc8901
# ╠═a1a3957a-8133-4930-a99f-461ff37c47f4
# ╟─87853196-5cda-4802-8d02-0666eed526cf
# ╠═9b8f60dd-890e-47ed-b1e3-bd812a2aa0b8
# ╟─3322ab74-2e74-43da-b56b-dc9050911b0c
# ╠═ca5fdc97-7533-4697-addc-02798bc0e5ae
# ╠═4e4da02b-2890-4bd4-a562-a528a565eebe
# ╟─cac8afd1-23c9-41eb-a9cf-aac803fd5289
# ╠═14bd5dc0-e515-40d3-aee2-8348a9c74100
# ╠═d8487d5e-8bab-40f2-a8df-4d27be2fbb77
# ╠═32eeb4e9-f7d1-4aac-b11b-c866a4f70bf7
# ╠═55f2750a-bbf8-448f-b900-57b37e235ba5
# ╠═70e2ad4a-b642-4fc8-a45f-df29a3231167
# ╠═229ebc35-f22f-4aaa-8ca6-4781895f75b8
# ╠═c370e0a7-3bdf-4f98-909a-76784199fbbe
# ╟─cbcec3a9-fdd6-465d-8d85-207feb6b78be
# ╟─df26309b-bb5a-4af8-98b6-fa1bf3accd8d
# ╟─0e56e1cd-4fe6-49fd-8a75-8cf1af8cb7f5
