using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, DataFramesMeta, CSV, MultivariateStats
#Worked with Ahmed and Aleeze
include("lgwt.jl")
function ps8()
#1
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2020/master/ProblemSets/PS8-factor/nlsy.csv"
	df = CSV.read(HTTP.get(url).body, DataFrames)

	rgrsn = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr), df)
	print(rgrsn)
    #print(coef(rgrsn))

#2
asvab=convert(Matrix,df[:,r"asvab"])
	correlation=cor(asvab)
	#for all variables:
	#cor(convert(Matrix,df))

	println(correlation)

#3
2rgrsn = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr+asvabAR+asvabCS+asvabMK+asvabNO+asvabPC+asvabWK), df)
	print(2rgrsn)
#These serve as proxies for the response. Keeping them results in measurement error;however, leaving them out causes omitted variable bias. I would believe they're all correlated to each other?

#4
M = fit(PCA, asvabMat'; maxoutdim=1)

asvabPCA = MultivariateStats.transform(M, asvabMat')
print(asvabPCA)

#5
M = fit(FactorAnalysis, asvabMat'; maxoutdim=1)
	asvabFCTRA = MultivariateStats.transform(M, asvabMat')

	print(asvabFCTRA)

#6

#Question 6
#I'm sorry I ran into a bit of trouble on this problem. I had to eventually be done.
	z=rand(Normal(0,1),size(df,1))
	A=df[:logwage]
	X=df[:,[:black,:hispanic,:female,:schoolt,:gradHS,:grad4yr]]
	asvab=df[:,[:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]]

	function loglike(sigma,m,z,y,x,asv)
	    S=size(A,1)
		T=size(asv,2)
		
		rez=zeros(S,T)
		for i=1:T
			rez[:,i]=residuals(lm(@formula(x1 ~ black + hispanic + female +x1_1),hcat(x,asvab[i],z,makeunique=true)))
		end
	   
		w=residuals(lm(@formula(x1 ~ black + hispanic + female + schoolt + gradHS + grad4yr+x1_1), hcat(x,y,z,makeunique=true)))
		
		return 
	end
end
ps8()