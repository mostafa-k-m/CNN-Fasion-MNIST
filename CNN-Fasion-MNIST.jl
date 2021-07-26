### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ 99ee4fa6-a080-11eb-1df7-1b2a6abf52a7
begin 
	using Markdown
	using InteractiveUtils
	using Flux, PlutoUI, Statistics, MLDatasets, Images
	using Flux.Data:DataLoader 
	using Flux: @epochs, onehotbatch, onecold, logitcrossentropy, throttle, unsqueeze
	using Random:shuffle
	using StatsBase: countmap, proportionmap
	using IterTools:ncycle
	using CUDA
	using Plots
	Plots.PlotlyBackend()
end

# ╔═╡ f3ce6f64-6e2b-43fb-aa08-1cd9b3ca91cf
begin
		using DarkMode
		DarkMode.enable(theme="monokai")
end

# ╔═╡ 9ddb04c6-6831-458e-a1ea-da0e5a079d9a
md"# Fashion MNIST training"

# ╔═╡ 1065d8b4-a1a5-473b-b12e-79d103ef2d09
begin
	train_x_, train_y_i = FashionMNIST.traindata(Float32)
	
	test_x_, test_y_i = FashionMNIST.testdata(Float32)
	
	train_y, test_y = onehotbatch(train_y_i, 0:9) |> gpu, onehotbatch(test_y_i, 0:9) |> gpu
	
	train_x = reshape(train_x_, :, size(train_x_)[3]) |> gpu
	test_x = reshape(test_x_, :, size(test_x_)[3])  |> gpu
	
end

# ╔═╡ b03a940e-cf96-4d25-a398-679bb36b9c37
md"We can look at few samples. Note that I do `1. .- tensor` to change the background to white"

# ╔═╡ 23214778-6693-4124-8e6d-5df995644dbe
begin
	class_dic = Dict(
		0 => "T-shirt/top", 
		1 => "Trouser", 
		2 => "Pullover", 
		3 => "Dress", 
		4 => "Coat", 
		5 => "Sandal", 
		6 => "Shirt", 
		7 => "Sneaker", 
		8 => "Bag", 
		9 => "Ankle boot"
	) 
	md"I use the above table to construct a dictionary to make labels to descriptions. Stored in the `class_dic` variable"
end

# ╔═╡ 836dc798-a15d-424a-b882-d91b02418017
classDict = Dict(
	0 => "T-shirt/top",
	1 => "Trouser",
	2 => "Pullover",
	3 => "Dress",
	4 => "Coat",
	5 => "Sandal",
	6 => "Shirt",
	7 => "Sneaker",
	8 => "Bag",
	9 => "Ankle boot"
)


# ╔═╡ 4e34013b-f272-4726-b795-ed9d2e41d61e
md"# Question 1: Defining the model"

# ╔═╡ 135d3c1c-a853-4afc-b045-eb34ceda43b7
model = Chain(
    Flux.Dense(784, 300, relu),
    Flux.Dense(300, 100, relu),
    Flux.Dense(100, 10),
    softmax
) |> gpu

# ╔═╡ 624105c2-096a-45e2-a62d-1fd0f4aa6aed
md"# Question 2"

# ╔═╡ 047152aa-283d-485a-8786-645514ab9f08
md"## Defining Loss Function"

# ╔═╡ 9e61ca71-60e8-4574-a49c-18fb38e05952
function loss(x, y)
	y_predict = model(x)
	return Flux.Losses.crossentropy(y_predict, y)
end

# ╔═╡ 3a1bc841-ad0a-42fc-a43e-24a726b56243
md"## Defining Accuracy Function"

# ╔═╡ fd968d0e-1e0b-4834-98e5-d2e0de4e51dd
function accuracy(x, y) 
	x = Flux.onecold(model(x))
	y = Flux.onecold(y)
	return mean(x .== y)
end


# ╔═╡ 1fd9bb58-a906-45db-88c3-c5f5e831e483
md"## Instatiate an ADAM optimizer"

# ╔═╡ e52bea6d-ca99-4bf9-a6ae-2b6d7e15d9fb
opt = Flux.Optimise.ADAM()

# ╔═╡ dafd28a4-aa40-4bbf-8459-8462ff8cd89b
md"# Question 3"

# ╔═╡ d7eb301a-cdea-4fbb-a179-eca4297d933f
md"## Defining badIdx Function"

# ╔═╡ 323f2325-c642-4920-87be-c1739fc939da
md"We will start by defining a function called getClass; this function is supposed to return the class name of a test datapoint or a test datalabel. We ended up having to define two methods for the same function to handle both test datapoints and labels."

# ╔═╡ bcbf53b3-4d59-43c5-b32f-916f7ccb8ead
function getClass(tensorIn, model)
	classIx = Flux.onecold(model(tensorIn), 0:9)
	return classDict[classIx]
end

# ╔═╡ d3fa55a4-5030-4646-8b73-5a10b6213a6e
function getClass(tensorIn::Flux.OneHotArray{UInt32,10,0,1,UInt32})
	classIx = Flux.onecold(tensorIn, 0:9)
	return classDict[classIx]
end

# ╔═╡ 8df24e6b-011c-412b-b39c-3850b51cf455
md"We used Our implementation of getClass to construct our badIdx function"

# ╔═╡ 0b269a43-1a55-4735-9750-db09f2bde59d
function badIdx(test_x, test_y, model)
	n = size(test_x)[2]
	doesNotMatch = []
	for i in 1:n
		trueClass = getClass(test_y[:,i])
		if getClass(test_x[:,i], model) != trueClass
			push!(doesNotMatch, i)
		end
	end
	return doesNotMatch
end


# ╔═╡ 59419dbd-2c35-4fdc-b690-802f4b6d11e1
md"# Question 4"

# ╔═╡ c3ee62cf-a3ff-4615-b44a-46c70a5d18c5
md"Creating the model parameters"

# ╔═╡ 0239bfd6-57e6-4d85-9547-e3501ca9985f
parameters = Flux.params(model)

# ╔═╡ 9d735b69-ba12-4b71-ab3b-5f8df56480ce
md"Creating dataloaders for the test and training datasets"

# ╔═╡ e7ed7470-a96a-4b0b-9ab2-6c496254753b
begin
	train_loader = Flux.Data.DataLoader((train_x, train_y), batchsize=50, shuffle=true)
	test_loader = Flux.Data.DataLoader((test_x, test_y), batchsize=50, shuffle=true)
end

# ╔═╡ ca07e8bf-c119-4e90-89a6-910765b35ff3
md"Defining a callback function, which prints to cunsole the loss and accuracy on the validation dataset when triggered"

# ╔═╡ 18b9d12e-66db-4f5b-868f-b7e97de2533d
function callBackEvaluation()
	lossVal = loss(test_x, test_y)
	accuracyVal = accuracy(test_x, test_y)
	@show(lossVal, accuracyVal)
end

# ╔═╡ 9fa4445d-1d81-41ae-a0a0-c2fdfd5c2914
md"Training the Model and Using ncycle and DataLoader. We specified that the callback funtion is to be triggered every 10 seconds"

# ╔═╡ b0b524c0-e742-43fe-8682-9ad53ac6c21a
Flux.Optimise.train!(
	loss,
	parameters,
	ncycle(train_loader, 10),
	opt,
	cb=Flux.throttle(callBackEvaluation, 10)
)

# ╔═╡ 6b243aef-5a68-4f93-b1ac-5d64833c8636
md"# Question 5: Calculating Classes Error Rates"

# ╔═╡ fd58bc7e-c17e-41d7-80c2-09bedbb0973d
md"We defined a function called getClassMismatchCount; which uses a given model to classify the validation dataset, and then compares its predictions to the given validation labels.

The function will output the number of misclassified pictures for each class, sorted by from worst to best performing"

# ╔═╡ 8d7f6d60-0074-479a-aaa9-610046f3fe73
function getClassMismatchCount(test_x, test_y, model)
	n = size(test_y)[2]
	classMismatchCount = Dict()
	predictions = Flux.onecold(model(test_x))
	for i in 1:n
		trueClass = getClass(test_y[:,i])
		if classDict[predictions[i]-1] != trueClass
			if haskey(classMismatchCount, trueClass)
				classMismatchCount[trueClass] += 1
    			else
				classMismatchCount[trueClass] = 1
			end
		end
	end
	mismatchCountVector = collect(classMismatchCount)
	return sort(mismatchCountVector, by=x -> x[2], rev=true)
end

# ╔═╡ a5de120c-3b4d-4744-b903-dd64b7e666f5
md"Next we created a function called getClassErrorRates, which uses the output from getClassMismatchCount and devides each class misclassified picture count by the number of datapoints in the class. the function returns the error rate for each class sorted from worst to best performing"

# ╔═╡ b63b3424-34aa-4795-87e1-60d3340ab511
classCounts = Dict([(classDict[i], count(x -> x == i, test_y_i)) for i in 0:9])

# ╔═╡ f01b1008-a244-46b6-aec7-b8d6ae87926d
function getClassErrorRates(mismatchCountVector)
	classErrorRates = Dict()
	for pair in mismatchCountVector
		classErrorRates[pair[1]] = pair[2] / classCounts[pair[1]]
	end
	classErrorRatesVector = collect(classErrorRates)
	return sort(classErrorRatesVector, by=x -> x[2], rev=true)
end

# ╔═╡ 2e18f690-423b-49ef-a0f1-ac23e2dcd37f
md"Finally we defined a function called analyzeMisclassification, this function will tell us for each class and identifies what wrong classes the model had predicted for it, and for each of these wrong classes, the function will calculate the percentage of the respective wrong class compared to the total number of misclassification.

In this way we will understand how the model misclassifies some items of clothing as others."

# ╔═╡ de60f2ad-bac2-4aba-b72a-a35420603275
function analyzeMisclassification(test_x, test_y, model)
	n = size(test_y)[2]
	classMismatch = Dict()
	classMismatchCount = Dict()
	predictions = Flux.onecold(model(test_x))
	for i in 1:n
		trueClass = getClass(test_y[:,i])
		predictedClass = classDict[predictions[i]-1]
		if predictedClass != trueClass
			if ~haskey(classMismatch, trueClass)
				classMismatch[trueClass] = Dict()
				classMismatchCount[trueClass] = 1
    			else
				classMismatchCount[trueClass] += 1
			end
			if haskey(classMismatch[trueClass], predictedClass)
				classMismatch[trueClass][predictedClass] += 1
    			else
				classMismatch[trueClass][predictedClass] = 1
			end
		end
	end
	for (key, _) in classMismatch
		for (key₂, value) in classMismatchCount
			if haskey(classMismatch[key], key₂)
				classMismatch[key][key₂] /= classMismatchCount[key]
			end
		end
		classMismatch[key] = collect(classMismatch[key])
		classMismatch[key] = sort(classMismatch[key], by=x -> x[2], rev=true)
	end
	return classMismatch
end

# ╔═╡ b5a56a9c-a405-456f-8e8f-8d35b2bce28b
md"Testing getClass"

# ╔═╡ 1096ca18-7a86-4448-bce5-2247c782c5cb
prediction = getClass(test_x[:,1], model)

# ╔═╡ c120d17d-a209-48ca-9cc5-032f4a822352
actual = getClass(test_y[:,1])

# ╔═╡ 5c998aaa-9ada-42db-af7d-88d388c977b0
md"Calculating the final accuracy of the MLP network"

# ╔═╡ e98b90e3-feca-4979-8ada-f1942dffd37a
finalAccuracy = accuracy(test_x, test_y)

# ╔═╡ 6bd03c2f-e101-477c-b6bf-9704965b9fe6
md"running badIdx on the resuts of the MLP"

# ╔═╡ 8426de5b-a487-4afc-bf1b-775f1ba9fa8e
badIndices = badIdx(test_x, test_y, model)

# ╔═╡ 3fd2addb-9c9b-4953-8e52-67d3a52c7d17
classMisMatchCounts = getClassMismatchCount(test_x, test_y, model)

# ╔═╡ c022f144-b278-4a97-ac62-78c7bc40d298
md"Running getClassMismatchCount on the results of the MLP.

We see that the top 3 misclassified classes are in order:
1. Shirt at 35.3% accuracy
2. Coat  at 28.2% accuracy
3. Pullover at 15.2% accuracy"

# ╔═╡ 5c9fb774-0d72-4191-a405-80d5c9ec0e90
classMisMatchRates = getClassErrorRates(classMisMatchCounts)

# ╔═╡ 8c59b031-be6b-4104-9a72-8126e4200ea3
md"# Question 6: Analyzing Class Error Rates"

# ╔═╡ 8f52c0cf-d7b1-45cb-a62a-352723d0e38d
md"We ran analyzeMisclassification on the results of the MLP to understand the mistakes that the model made"

# ╔═╡ 147e90b7-aa66-40ac-8b2e-c561c96d6846
classMisMatchAnalysis = analyzeMisclassification(test_x, test_y, model)

# ╔═╡ b0530aec-084f-4ff2-9636-64b2715617c7
md"Here we defined some functions to do plots that we need to visualize the model results"

# ╔═╡ 836d6dde-e19f-4e26-8cc8-a34e988052c5
begin
	function PlotPair(pairs, label, title)
		names = []
		values= []
		for pair in pairs
			push!(names, pair[1])
			push!(values, pair[2])
		end
		p = bar(values, xticks=(1:10, names), label=label, title=title, xrotation=45)
		return p
	end
end

# ╔═╡ 5b8e5835-389d-4f30-b4f4-aac3bd2eece1
begin
	function PlotDict(classMisMatchAnalysis)		
		nFigures = length(classMisMatchAnalysis)
		l = @layout [grid(ceil(nFigures/2) |> Int, 2)]
		bars = []
		for (key, value) in classMisMatchAnalysis
			names = []
			values= []
			for pair in value
				push!(names, pair[1])
				push!(values, pair[2])
			end
			bar_ = bar(values, xticks=(1:10, names), label="mismatch rate", title=key, xrotation=45)
			push!(bars, bar_)
		end
		p = plot(bars... ,layout = (ceil(nFigures/2) |> Int, 2), size=(1000,1500))
		return p
	end
end

# ╔═╡ 2c1a3d88-67b8-447a-9df4-6ff02e85a7e6
md"in this plot we visualized the error rate for each class. We notice that Shirt, Coat and T-shirt/top are the top three misclassified classes.

We also notice that articles of clothing that are worn on the upper body e.g. Shirt, Coat, T-shirt/top, pullover and dress, have much higher error rates compared to other classes, suggesting that they are often misclassified as each other. We will analyze that further shortly."

# ╔═╡ a6f2c582-52fb-410a-a5f4-c285482b3e8e
PlotPair(classMisMatchRates, "Mismatch Rate", "Error Rate Per Class")

# ╔═╡ f24da486-076e-46b2-87df-235d17fb1b8f
md"In this plot we are looking at each individual class in the validation dataset, and identifying what percentage of its misclassification instances were maped to what wrong class.

For example, We see that the misclassified sneakes, were identified as either sandals or ankle boots by the model.. etc."

# ╔═╡ 62fc7fdc-1bbe-4b81-b842-22f77af137f7
PlotDict(classMisMatchAnalysis)

# ╔═╡ 90165a56-e947-47fb-bb8c-c1dedba180b2
md"# Question 6 (repeated): Building a Custom Training Loop"

# ╔═╡ e559d08b-7d64-4b9f-ac92-c608b8494ed0
md"We made an identical MLP model to train with the custom training loop"

# ╔═╡ 6d7f0581-73af-4c64-8fe8-97a610264464
model_cust = Chain(
    Flux.Dense(784, 300, relu), # Flattened input Image -> 300 Nodes Hidden Layer 1
    Flux.Dense(300, 100, relu), # 300 Nodes Hidden Layer 1-> 100 Nodes Hidden Layer 2
    Flux.Dense(100, 10), # 100 Nodes Hidden Layer 2-> 10 Nodes Output
    softmax # Softmax for Classification
) |> gpu

# ╔═╡ 9890e496-6b10-42a8-b311-838c061e2db2
function accuracy_cust(x, y, model_cust) 
	x = Flux.onecold(model_cust(x))
	y = Flux.onecold(y)
	return mean(x .== y)
end

# ╔═╡ 460d5afd-715e-497d-b374-f3f93f491845
md"Defining the custom training loop. This loop will train a model for 10 epochs and will output the validation loss and the accuracy after each epoch."

# ╔═╡ d7e0895f-7a29-4063-b13b-744ab71be228
function customTrainingLoop(model_cust, trainLoader, test_x, test_y)
	function loss_cust(x, y)
		y_predict = model_cust(x)
		return Flux.Losses.crossentropy(y_predict, y)
	end

	opt_cust = Flux.Optimise.ADAM()
	params_cust = Flux.params(model_cust)

	for _ in 1:10
		for (train_point, test_point) in trainLoader
			grads = gradient(params_cust) do
				loss_cust(train_point, test_point)
			end 
			Flux.Optimise.update!(opt_cust, params_cust, grads)
		end
		@show loss_cust(test_x, test_y) , accuracy_cust(test_x, test_y, model_cust)
	end
end

# ╔═╡ 68c2a1f5-bf7a-499e-b82b-7006377be589
md"training the second MLP with the custom training loop and getting its final accuracy"

# ╔═╡ 16a41d4e-66e7-4d67-804a-79157f75f8ee
customTrainingLoop(model_cust, train_loader, test_x, test_y)

# ╔═╡ e3cd0de9-4666-49b7-b208-4ed6a6b6ec24
accuracy_cust(test_x, test_y, model_cust)

# ╔═╡ b7a6a5c1-866f-4a4a-8efa-ea5b14473540
md"# Question 7: Creating a CNN Model"

# ╔═╡ a2ed369d-4c57-4644-b34e-7c3518d652b7
md"For the CNN model we have to reshape the dataset before we can start training the model"

# ╔═╡ 9ca840c9-0181-422f-a51c-a501a2edc14e
train_x_conv = reshape(train_x_, 28, 28, 1, :) |> gpu

# ╔═╡ 47780a1e-5606-4650-a530-a25bbac2a642
train_loader_conv = Flux.Data.DataLoader((train_x_conv, train_y), batchsize=50, shuffle=true)

# ╔═╡ 60c739e5-b91f-4488-8a37-783fcaebdbc6
test_x_conv = reshape(test_x_, 28, 28, 1, :)  |> gpu

# ╔═╡ 4c9dd198-9c29-47d2-9219-232f879846da
md"Here we defined a CNN model as described in the assignment questions"

# ╔═╡ 04c849be-0373-4349-891b-fdc60637f177
model_conv = Chain(
	Conv((5, 5), 1=>6, pad=SamePad(), relu),
	MaxPool((2, 2)),
	Conv((5, 5), 6=>16, pad=SamePad(), relu),
	MaxPool((2, 2)),
	flatten,
    Dense(784, 10),
    softmax,
) |> gpu

# ╔═╡ f9bfdece-9dd5-4b3a-9813-933a2cd714e2
md"We trained the CNN model with our custom training loop. And tested its final accuracy"

# ╔═╡ 2bd4725c-2b29-4571-b248-2878e888269a
customTrainingLoop(model_conv, train_loader_conv, test_x_conv, test_y)

# ╔═╡ 2093efe1-fb9a-4bed-a973-6f7578441e81
accuracy_cust(test_x_conv, test_y, model_conv)

# ╔═╡ 0203e440-ce41-417e-88f5-045c295a699b
md"# Question 8: Improving the CNN Model Through Architectural Changes"

# ╔═╡ 17d3f6b8-4d8f-4c8a-a19e-3b3c859061da
md"We slightly modified the previous CNN architecure as followes. We will use the LeNet Architecture"

# ╔═╡ 2fd460fd-ab6c-4734-8c4c-0bad6adcd8cf
model_conv_2 = Chain(
    # First convolution, operating upon a 28x28 image
    Conv((3, 3), 1 => 16, pad=(1, 1), relu),
    MaxPool((2, 2)),

    # Second convolution, operating upon a 14x14 image
    Conv((3, 3), 16 => 32, pad=(1, 1), relu),
    MaxPool((2, 2)),

    # Third convolution, operating upon a 7x7 image
    Conv((3, 3), 32 => 32, pad=(1, 1), relu),
    MaxPool((2, 2)),

    # `Flux.flatten` will make the pictures (3, 3, 32, N)
    flatten,
    Dense(288,  128, relu), 
	Dense(128, 64, relu),
	Dense(64, 10),
	softmax
	) |> gpu

# ╔═╡ d125bde9-1083-4406-b986-69bd7a4e8749
md"We traind the new CNN model with our custom trainig loop"

# ╔═╡ 684de755-413c-4d8e-8dce-f8cc86737ef4
customTrainingLoop(model_conv_2, train_loader_conv, test_x_conv, test_y)

# ╔═╡ 68482d3f-5884-44dc-9f30-4034ddf7504d
md"We can see that this model performs better at more than 90% accuracy"

# ╔═╡ c36ff13b-d261-455e-a404-de5ecc83d41f
accuracy_cust(test_x_conv, test_y, model_conv_2)

# ╔═╡ 50e9867c-1c9f-4e19-848d-f8742e89a867
md"We will do the same analysis we did on the results of the first MLP model"

# ╔═╡ 3996ca19-00ae-457e-b057-7721dbaf2440
classMisMatchCountsConv = getClassMismatchCount(test_x_conv, test_y, model_conv_2)

# ╔═╡ 84ae8979-ba0a-4712-a73b-753aeb183457
md"We again see that the top misclassified classes are all upperbody clothing items. however the top three misclassified items here are different:
1. Shirt at 25% error Rate.
2. T-shirt/top at 15.7% error Rate.
3. Coat 14.1% error Rate."

# ╔═╡ 37847a94-98be-4f16-aa7a-c9ccb18afe1a
classMisMatchRatesConv = getClassErrorRates(classMisMatchCountsConv)

# ╔═╡ 34973ae2-3ddd-4df6-9f2f-982bca3a6246
PlotPair(classMisMatchRatesConv, "Mismatch Rate", "Error Per Class")

# ╔═╡ 42fba4b7-afc5-4388-b677-b89c82f29b21
classErrorAnalysisConv2 = analyzeMisclassification(test_x_conv, test_y, model_conv_2)

# ╔═╡ b98672ed-82a1-48b2-8e9f-ad0ef3504c8d
PlotDict(classErrorAnalysisConv2)

# ╔═╡ 1564421b-d3b1-4c43-a60a-c7437797123a
md"# Question 9"

# ╔═╡ 45d66e23-80c5-4ffe-b16b-22f730a1c01a
md"yes, the top 3 mmisclassified claasses changed across the models we trained, they also changed if we trained the same model"

# ╔═╡ df26309b-bb5a-4af8-98b6-fa1bf3accd8d
md"""
## Assignment Questions 


!!! question "Question 1 (4 pts)"

    Using Flux `Chain` build a Dense MLP NN, with following layers in sequence 784, 300, 100, 10. Run the output of the lastad layer through a `softmax` function.     

!!! question "Question 2 (2 pts)"

    a) Define a loss funtion based on `crossentropty`

	b) Instatiate an `ADAM` optimizer 

	c) Define an `accuracy` function that could runn over the whole dataset (Hint: make use of `onecold` and `mean`) 


!!! question "Question 3 (3 pts)"

    Define a funciton `badIdx` that filter for the indices where the model failes to classify correctly. 

!!! question "Question 4 (3 pts)"

    Make use of `nycle` and `DataLoader` and `Flux.train!` to train the model with `batchsize` of 50 and 10 epochs (Hint: see [docs](https://fluxml.ai/Flux.jl/stable/data/dataloader/)). Use a callback function, `cb` to the `accuracy` and `loss` every 10 seconds.  

!!! question "Question 5 (4 pts)"

    Which top 3 classes did the Dense MLP NN model struggle with the most? Are ther error rates uniform across the classes? If the repeat the training, it the top three misclassified classes 3×3×20×1

!!! question "Question 6 (4 pts)"

    Which top 3 classes did the Dense MLP NN model struggle with the most? Are ther error rates uniform across the classes? If the repeat the training, it the top three misclassified classes 

!!! question "Question 6 (3 pts)"

    Build you own *custom training loop* with same parameters. Check that it works just like in question 4. 


!!! question "Question 7 (5 pts)"

    Construct a convolutional model with following architecture 

		1. Conv with a (5,5) kernel mapping to **6** feature maps, with `relu`,  same paddig 
		2. 2x2 Max pool 
		3. Conv with a (5,5) kernel mapping to **16** feature maps, with `relu`,  same paddig 
		4. 2x2 Max pool
		5. Appropriatly sized Dense layer with 10 outputs
		6. A `softmax` layer
	
	Do the training using a *custom training loop* 

!!! question "Question 8 (3 pts)"

	Do changes in the convnet  architecture to get a better accuracy. 

!!! question "Question 9 (3 pts)"

	Did the top 3 classes in Question 6 change with the use of the networks in Questions 8 or 9?  










"""

# ╔═╡ Cell order:
# ╟─9ddb04c6-6831-458e-a1ea-da0e5a079d9a
# ╠═99ee4fa6-a080-11eb-1df7-1b2a6abf52a7
# ╠═f3ce6f64-6e2b-43fb-aa08-1cd9b3ca91cf
# ╟─1065d8b4-a1a5-473b-b12e-79d103ef2d09
# ╟─b03a940e-cf96-4d25-a398-679bb36b9c37
# ╟─23214778-6693-4124-8e6d-5df995644dbe
# ╟─836dc798-a15d-424a-b882-d91b02418017
# ╟─4e34013b-f272-4726-b795-ed9d2e41d61e
# ╠═135d3c1c-a853-4afc-b045-eb34ceda43b7
# ╟─624105c2-096a-45e2-a62d-1fd0f4aa6aed
# ╟─047152aa-283d-485a-8786-645514ab9f08
# ╠═9e61ca71-60e8-4574-a49c-18fb38e05952
# ╟─3a1bc841-ad0a-42fc-a43e-24a726b56243
# ╠═fd968d0e-1e0b-4834-98e5-d2e0de4e51dd
# ╟─1fd9bb58-a906-45db-88c3-c5f5e831e483
# ╠═e52bea6d-ca99-4bf9-a6ae-2b6d7e15d9fb
# ╟─dafd28a4-aa40-4bbf-8459-8462ff8cd89b
# ╟─d7eb301a-cdea-4fbb-a179-eca4297d933f
# ╟─323f2325-c642-4920-87be-c1739fc939da
# ╠═bcbf53b3-4d59-43c5-b32f-916f7ccb8ead
# ╠═d3fa55a4-5030-4646-8b73-5a10b6213a6e
# ╟─8df24e6b-011c-412b-b39c-3850b51cf455
# ╠═0b269a43-1a55-4735-9750-db09f2bde59d
# ╟─59419dbd-2c35-4fdc-b690-802f4b6d11e1
# ╟─c3ee62cf-a3ff-4615-b44a-46c70a5d18c5
# ╠═0239bfd6-57e6-4d85-9547-e3501ca9985f
# ╟─9d735b69-ba12-4b71-ab3b-5f8df56480ce
# ╠═e7ed7470-a96a-4b0b-9ab2-6c496254753b
# ╠═ca07e8bf-c119-4e90-89a6-910765b35ff3
# ╠═18b9d12e-66db-4f5b-868f-b7e97de2533d
# ╟─9fa4445d-1d81-41ae-a0a0-c2fdfd5c2914
# ╠═b0b524c0-e742-43fe-8682-9ad53ac6c21a
# ╟─6b243aef-5a68-4f93-b1ac-5d64833c8636
# ╟─fd58bc7e-c17e-41d7-80c2-09bedbb0973d
# ╠═8d7f6d60-0074-479a-aaa9-610046f3fe73
# ╟─a5de120c-3b4d-4744-b903-dd64b7e666f5
# ╠═b63b3424-34aa-4795-87e1-60d3340ab511
# ╠═f01b1008-a244-46b6-aec7-b8d6ae87926d
# ╟─2e18f690-423b-49ef-a0f1-ac23e2dcd37f
# ╠═de60f2ad-bac2-4aba-b72a-a35420603275
# ╟─b5a56a9c-a405-456f-8e8f-8d35b2bce28b
# ╠═1096ca18-7a86-4448-bce5-2247c782c5cb
# ╠═c120d17d-a209-48ca-9cc5-032f4a822352
# ╟─5c998aaa-9ada-42db-af7d-88d388c977b0
# ╠═e98b90e3-feca-4979-8ada-f1942dffd37a
# ╟─6bd03c2f-e101-477c-b6bf-9704965b9fe6
# ╠═8426de5b-a487-4afc-bf1b-775f1ba9fa8e
# ╠═3fd2addb-9c9b-4953-8e52-67d3a52c7d17
# ╟─c022f144-b278-4a97-ac62-78c7bc40d298
# ╠═5c9fb774-0d72-4191-a405-80d5c9ec0e90
# ╟─8c59b031-be6b-4104-9a72-8126e4200ea3
# ╟─8f52c0cf-d7b1-45cb-a62a-352723d0e38d
# ╠═147e90b7-aa66-40ac-8b2e-c561c96d6846
# ╟─b0530aec-084f-4ff2-9636-64b2715617c7
# ╠═836d6dde-e19f-4e26-8cc8-a34e988052c5
# ╠═5b8e5835-389d-4f30-b4f4-aac3bd2eece1
# ╟─2c1a3d88-67b8-447a-9df4-6ff02e85a7e6
# ╠═a6f2c582-52fb-410a-a5f4-c285482b3e8e
# ╟─f24da486-076e-46b2-87df-235d17fb1b8f
# ╠═62fc7fdc-1bbe-4b81-b842-22f77af137f7
# ╠═90165a56-e947-47fb-bb8c-c1dedba180b2
# ╟─e559d08b-7d64-4b9f-ac92-c608b8494ed0
# ╠═6d7f0581-73af-4c64-8fe8-97a610264464
# ╠═9890e496-6b10-42a8-b311-838c061e2db2
# ╟─460d5afd-715e-497d-b374-f3f93f491845
# ╠═d7e0895f-7a29-4063-b13b-744ab71be228
# ╟─68c2a1f5-bf7a-499e-b82b-7006377be589
# ╠═16a41d4e-66e7-4d67-804a-79157f75f8ee
# ╠═e3cd0de9-4666-49b7-b208-4ed6a6b6ec24
# ╟─b7a6a5c1-866f-4a4a-8efa-ea5b14473540
# ╟─a2ed369d-4c57-4644-b34e-7c3518d652b7
# ╠═9ca840c9-0181-422f-a51c-a501a2edc14e
# ╠═47780a1e-5606-4650-a530-a25bbac2a642
# ╠═60c739e5-b91f-4488-8a37-783fcaebdbc6
# ╟─4c9dd198-9c29-47d2-9219-232f879846da
# ╠═04c849be-0373-4349-891b-fdc60637f177
# ╟─f9bfdece-9dd5-4b3a-9813-933a2cd714e2
# ╠═2bd4725c-2b29-4571-b248-2878e888269a
# ╠═2093efe1-fb9a-4bed-a973-6f7578441e81
# ╟─0203e440-ce41-417e-88f5-045c295a699b
# ╟─17d3f6b8-4d8f-4c8a-a19e-3b3c859061da
# ╠═2fd460fd-ab6c-4734-8c4c-0bad6adcd8cf
# ╟─d125bde9-1083-4406-b986-69bd7a4e8749
# ╠═684de755-413c-4d8e-8dce-f8cc86737ef4
# ╟─68482d3f-5884-44dc-9f30-4034ddf7504d
# ╠═c36ff13b-d261-455e-a404-de5ecc83d41f
# ╟─50e9867c-1c9f-4e19-848d-f8742e89a867
# ╠═3996ca19-00ae-457e-b057-7721dbaf2440
# ╟─84ae8979-ba0a-4712-a73b-753aeb183457
# ╠═37847a94-98be-4f16-aa7a-c9ccb18afe1a
# ╠═34973ae2-3ddd-4df6-9f2f-982bca3a6246
# ╠═42fba4b7-afc5-4388-b677-b89c82f29b21
# ╠═b98672ed-82a1-48b2-8e9f-ad0ef3504c8d
# ╟─1564421b-d3b1-4c43-a60a-c7437797123a
# ╟─45d66e23-80c5-4ffe-b16b-22f730a1c01a
# ╟─df26309b-bb5a-4af8-98b6-fa1bf3accd8d
