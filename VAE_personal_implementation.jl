## I try to implement again a simple VAE using Julia
## There is a tutorial with a working VAE in the Flux documentation
## (https://github.com/jessebett/model-zoo/blob/master/vision/mnist/vae.jl)
## I try to do mine in a different way

## I am using the original Binarized MNIST
## you can download it from http://www.dmi.usherb.ca/~larocheh/mlpython/_modules/datasets/binarized_mnist.html
## I assume the files are into a folder called "Original_MNIST_binarized"
## I am not passing a directory to the loading function (possibly a TODO)

using Flux, Statistics
using Flux: Tracker, throttle, params, binarycrossentropy, gradient, @epochs, testmode!, glorot_uniform
#flux function for update parameters
using Flux.Tracker: update!
using Base.Iterators: partition
using Juno: @progress
using MLDataUtils
using Images
include("utils.jl")

img(x) = Gray.(permutedims(reshape(x, 28, 28)))

function log_standard_gaussian(x)
    #return torch.sum(-0.5 * math.log(2 * math.pi) - x ** 2 / 2, dim=-1)
    #log_pdf =  -0.5f0 * log(2 * pi) .- x.^2 / 2
    log_pdf =  @. -0.5f0 * log(2 * pi) - x^2 / 2
    #@show(A)
    # alternatively, we can do
    #Norm = Normal(0)
    #A = log(pdf(Norm, x))
    #@show(A)
    return log_pdf
end

# function log_gaussian(x, mu, log_var)
#     std = exp.(log_var * 0.5f0)
#     #@show log_var
#     #@show std
#     Norm = Normal(mu, std)
#     A = log(pdf(Norm, x))
#     return A
# end

function log_gaussian(x, mu, log_var)
    log_pdf = - 0.5 * log(2 * pi) .- log_var / 2 - (x .- mu)^2 / (2 * exp.(log_var))
    #log_pdf = @. - 0.5 * log(2 * pi) - (log_var + eps(Float32))1 / 2 - (x - mu)^2 / (2 * exp(log_var))
    # print('Size log_pdf:', log_pdf.shape)
    return log_pdf
end


# function binary_cross_entropy(recon, x)
#     # we can also use "using Flux: binarycrossentropy"
#     # sum(binarycrossentropy.(recon, x))
#     s = @. x * log(recon + Float32(1e-8)) + (1 - x) * log(1 - recon + Float32(1e-8))
#     return -sum(s)
# end  THIS FUNCTION IS SUPER SLOW, MY GOODNESS! USE THE ONE PROVIDE BY FLUX

function reparametrization_trick(mu, log_var)
    z = mu + rand(Float32) * exp(log_var * 0.5)
    return z
end

function analytic_kl(mu, log_var)
    kl = (log_var - mu.^2 .+ 1.0 - exp.(log_var)) * 0.5
    return kl
end

## todo: I have to understand why the approaximate KL does not work
function _kl_divergence(z, mu_q, log_var_q, mu_p = nothing, log_var_p = nothing)
    # instead of computing it analytically, we can estimate it
    # using monte carlo methods
    qz = log_gaussian.(z, mu_q, log_var_q)
    ## we should do the same with p
    if isnothing(mu_p) & isnothing(log_var_p)
        pz = log_standard_gaussian.(z)
    else
        pz = log_gaussian.(z, mu_p, log_var_p)
    end
    kl = qz - pz
    return kl
end

const batch_size = 64
const latent_dim = 3
const hidden_dim = 400
const n_epochs = 200

# we can load the MNIST dataset
x_train, x_valid, x_test = load_binarized_mnist(batch_size, true)
example = x_train[:,1]
save("original2.png", img(example))

N = size(x_train, 2)
N_test = size(x_test, 2)
x_train_batch = [x_train[:,i] for i in Iterators.partition(1:N,batch_size)]
x_test_batch = [x_test[:,i] for i in Iterators.partition(1:N_test,batch_size)]
# we have to create the encoder
h, mu, log_var = Dense(28*28, hidden_dim, relu, initW = glorot_uniform, initb = zeros), Dense(hidden_dim, latent_dim, initW = glorot_uniform, initb = zeros), Dense(hidden_dim, latent_dim, initW = glorot_uniform, initb = zeros)
encoder(X) = (hidden_activation = h(X); (mu(hidden_activation), log_var(hidden_activation)))

# the decoder
decoder = Chain(Dense(latent_dim, hidden_dim, relu, initW = glorot_uniform, initb = zeros),
                Dense(hidden_dim, 28*28, sigmoid, initW = glorot_uniform, initb = zeros))

# we define the
# we have to define a method to samples
get_sample() = decoder(randn(Float32,latent_dim))
#function get_sample()
#    testmode!(decoder)
#    return decoder(randn(Float32,latent_dim))
#end
# now we have to define the callback --> used to observe the training process
function compute_loss(_data::Matrix)
    # we should pass our data into the encoder
    (_mu, _log_var) = encoder(_data)
    # then we use the reparametrization trick to sample z
    _z = reparametrization_trick.(_mu, _log_var)
    # we pass the z through the decoder to get the mean for each pixel
    mu_pixels = decoder(_z)

    # now we should compute the compute_loss
    likelihood = - sum(binarycrossentropy.(mu_pixels, _data))
    #@show likelihood
    kl = - sum(analytic_kl(_mu, _log_var))
    #kl = _kl_divergence(_z, _mu, _log_var)
    #©@show sum(kl)
    elbo = likelihood - kl
    loss = -elbo
    #@show loss

    #@show loss

    return loss, -likelihood, kl
end

opt = ADAM(3e-4)
ps = params(h, mu, log_var, decoder)

#
function train!()
    for ep in 1:n_epochs
        @info "Training Epoch $ep..."
        tot_recon = 0
        tot_kl = 0
        tot_loss = 0
        j = 0
        for data in x_train_batch
            #testmode!((h, mu, log_var, decoder), false)
            #@show j
            j += 1
            #we have to compute the loss of this
            (loss, _recon_err, _kl) = compute_loss(data)
            grad = Flux.Tracker.gradient(()-> loss/size(data,2), ps)
            update!(opt, ps, grad)
            tot_recon += _recon_err
            tot_kl += sum(_kl)
            tot_loss += loss
            #Flux.back!(l)<
            #Flux.Optimise._update_params!(opt, ps)
        end
        # we are at the end of the epochs
        println("Epoch: $ep, -ELBO: $(tot_loss/size(x_train,2)), recon err: $(tot_recon/size(x_train,2)), KL: $(tot_kl/size(x_train,2))")

        # ## TODO: understand how to sample during training, becuase at the moment it gets problems
        # (h, mu, log_var, decoder) = mapleaves(Flux.Tracker.data, ps)
        testmode!((h, mu, log_var, decoder), true)

        sample = img.([get_sample() for i = 1:64])
        @show(size(sample))
        grid_samples = vcat([hcat(imgs...) for imgs in partition(sample, 8)]...)
        save("samples/samples_from_random_epoch_$ep.png", grid_samples)

        example = [x_test[:,rand(1:N_test)] for i = 1:64]
        example2 = hcat(example...)
        grid_original = vcat([hcat(imgs...) for imgs in partition(img.(example), 8)]...)
        save("reconstructions/original_epoch_$ep.png", grid_original)
        m, lv = encoder(example2)
        zed2 = reparametrization_trick.(m,lv)
        recon = decoder(zed2)

        recon_to_save = img.([recon[:,i] for i = 1:64])
        grid_recon = vcat([hcat(imgs...) for imgs in partition(recon_to_save, 8)]...)
        save("reconstructions/reconstruction_epoch_$ep.png", grid_recon)
        testmode!((h, mu, log_var, decoder), false)
    end
end

train!()
#
#@epochs n_epochs train!()
# evalcb = throttle(() -> @show(ELBO(x_train[:, rand(1:N, batch_size)])/batch_size) , 30)
# @progress for i = 1:10
#   @info "Epoch $i"
#   Flux.train!(loss, ps, zip(x_train_batch), opt, cb=evalcb)
# end
(h, mu, log_var, decoder) = mapleaves(Flux.Tracker.data, (h, mu, log_var, decoder))
testmode!((h, mu, log_var, decoder), true)

sample = img.([get_sample() for i = 1:64])
@show(size(sample))
grid_samples = vcat([hcat(imgs...) for imgs in partition(sample, 8)]...)
save("samples/samples_from_random_epoch_$ep.png", grid_samples)

example = [x_test[:,rand(1:N_test)] for i = 1:64]
example2 = hcat(example...)
grid_original = vcat([hcat(imgs...) for imgs in partition(img.(example), 8)]...)
save("reconstructions/original_epoch_$ep.png", grid_original)
m, lv = encoder(example2)
zed2 = reparametrization_trick.(m,lv)
recon = decoder(zed2)

recon_to_save = img.([recon[:,i] for i = 1:64])
grid_recon = vcat([hcat(imgs...) for imgs in partition(recon_to_save, 8)]...)
save("reconstructions/reconstruction_epoch_$ep.png", grid_recon)