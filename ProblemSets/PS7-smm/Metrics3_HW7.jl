using Optim, SMM, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, DataFramesMeta, CSV
#Worked with Ahmed and Aleeze
function ps7()
#Q1
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.married.==1

function ols_gmm(alpha, X,y)
        g = y .- X*alpha
        J = g'*I*g
        return J
end
alphhat_optim = optimize(a -> ols_gmm(a, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-8, iterations=100_000))
println(alphhat_optim.minimizer)
# Checking work
bols = inv(X'*X)*X'*y

#2a


df = dropmissing(df, :occupation)
df[df.occupation.==8 ,:occupation] .= 7
df[df.occupation.==9 ,:occupation] .= 7
df[df.occupation.==10,:occupation] .= 7
df[df.occupation.==11,:occupation] .= 7
df[df.occupation.==12,:occupation] .= 7
df[df.occupation.==13,:occupation] .= 7

X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.occupation

function mlogit(alpha, X, y)
    
    K = size(X,2)
    J = length(unique(y))
    N = length(y)
    bigY = zeros(N,J)
    for j=1:J
        bigY[:,j] = y.==j
    end
    bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
    
    num = zeros(N,J)
    dem = zeros(N)
    for j=1:J
        num[:,j] = exp.(X*bigAlpha[:,j])
        dem .+= num[:,j]
    end
    
    P = num./repeat(dem,1,J)
    
    loglike = -sum( bigY.*log.(P) )
    
    return loglike
end

alpha_zero = zeros(6*size(X,2))
alpha_rand = rand(6*size(X,2))
alpha_true = [.1910213,-.0335262,.5963968,.4165052,-.1698368,-.0359784,1.30684,-.430997,.6894727,-.0104578,.5231634,-1.492475,-2.26748,-.0053001,1.391402,-.9849661,-1.398468,-.0142969,-.0176531,-1.495123,.2454891,-.0067267,-.5382892,-3.78975]
alpha_start = alpha_true.*rand(size(alpha_true))
println(size(alpha_true))

#2b

alpha_hat_optim = optimize(a -> mlogit(a, X, y), alpha_start, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
alpha_hat_mle = alpha_hat_optim.minimizer
println(alpha_hat_mle)

#2C

function logit_gmm(alpha, X, y)
    K = size(X,2)
    J = length(unique(y))
    N = length(y)
    bigY = zeros(N,J)
    for j=1:J
        bigY[:,j] = y.==j
    end
    bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
    
    num = zeros(N,J)
    dem = zeros(N)
    for j=1:J
        num[:,j] = exp.(X*bigAlpha[:,j])
        dem .+= num[:,j]
    end
    
    P = num./repeat(dem,1,J)
    
    loglike = -sum( bigY.*log.(P) )
    
    return loglike
end

alpha_hat_optim = optimize(a -> logit_gmm(a, X, y), alpha_true, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
println(alpha_hat_optim)

alpha_hat_optim = optimize(a -> logit_gmm(a, X, y), alpha_rand, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
println(alpha_hat_optim)

#The objective function looks globally concave. 

#3a
XR = rand(1000,2)
#3b
beta = [[(.4),(.1),(.5)] [(.5),(.2),(.4)]]]
#3c
p = XR*beta
#3d
eps = random(GeneralizedExtremeValue(0, 1, 0), 1000, 3)

GenY = zeros(1000,1)
for i in 1:size((XR),1)
    GenY[i,1] = argmax(p[i,:] + eps[i,:])
end

#4

MA = SMM.parallelNormal()
dc = SMM.history(MA.chains[1])
dc = dc[dc[:accepted].==true, :]
println(describe(dc))

#5, I'm sorry but I was very confused on this question. I tried mixing and matching with the code you gave in 
#the slides but I'm sure I'm missing something here.
function gmm_smm(θ, X, y, D)
    K = size(X,2)
    N = size(y,1)
    β = θ[1:end-1]
    σ = θ[end]
    if length(β)==1
        β = β[1]
    end
    # N+1 moments in both model and data
    gmodel = zeros(N+1,D)
    # data moments are just the y vector itself
    # and the variance of the y vector
    gdata  = vcat(y,var(y))
    #### !!!!!!!!!!!!!!!!!!!!!!!!!!!!! ####
    # This is critical!                   #
    Random.seed!(1234)                    #
    # You must always use the same ε draw #
    # for every guess of θ!               #
    #### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ###
    # simulated model moments
    for d=1:D
        ε = σ*randn(N)
        ỹ = X*β .+ ε
        gmodel[1:end-1,d] = ỹ
        gmodel[  end  ,d] = var(ỹ)
    end
    # criterion function
    err = vec(gdata .- mean(gmodel; dims=2))
    # weighting matrix is the identity matrix
    # minimize weighted difference between data and moments
    J = err'*I*err
    return J
end

alpha_hat_smm = optimize(a -> gmm_smm(a, X, y, 1000), alpha_true, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
println(alpha_hat_smm)
end

ps7()