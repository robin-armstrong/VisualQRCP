using LinearAlgebra
using Printf

struct QRCP
	Q::Matrix{Float64}
	R::Matrix{Float64}
	p::Vector{Int64}
end

Base.iterate(F::QRCP) = (F.Q, Val(:R))
Base.iterate(F::QRCP, ::Val{:R}) = (F.R, Val(:p))
Base.iterate(F::QRCP, ::Val{:p}) = (F.p, Val(:done))
Base.iterate(F::QRCP, ::Val{:done}) = nothing

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, F::QRCP)
	summary(io, F); println(io)
	println(io, "permutation:")
	show(io, mime, F.p)
	println(io, "\nQ factor:")
	show(io, mime, F.Q)
	println(io, "\nR factor:")
	show(io, mime, F.R)
end


"""
	visualQRCP(A ; tol = 0., silent = false)
	
Compute a Golub-Businger style QRCP of `A`, showing every step if `silent == false`. Terminate the algorithm when the largest remaining column norm is no greater than `tol`.
"""
function visualQRCP(A::Matrix{F} ; tol::Float64 = 0., silent::Bool = false) where {F <: Real}
	m, n = size(A)
	perm = Array{Int64}(range(1, n, n))
	Q = Matrix{Float64}(I(m))
	R = Matrix{Float64}(A)
	sqnorms = Vector{Float64}(undef, n)
	sqnorm_max = tol
	jmax = 0
	
	for j = 1:n
		sqnorms[j] = norm(R[:, j])^2
		if(sqnorms[j] > sqnorm_max)
			sqnorm_max = sqnorms[j]
			jmax = j
		end
	end
	
	k = 1
	
	while(k < min(m, n))
		if(!silent)
			# display the current state of the factorization
			run(Cmd(["clear"]))
			println("permutation | squared column norms")
			println("----------------------------------")
			
			for j = 1:n
				@printf("     %2d     |      %8.2e       \n", perm[j], sqnorms[j])
			end
			
			println("\nQ = ")
			display(Q)
			println("\nR = ")
			display(R)
			println("\n press enter to continue...")
			
			readline()
		end
		
		(sqnorm_max > tol) || break
		
		# swapping columns
		
		tmp = perm[k]
		perm[k] = perm[jmax]
		perm[jmax] = tmp
		
		tmp = sqnorms[k]
		sqnorms[k] = sqnorms[jmax]
		sqnorms[jmax] = tmp
		
		tmp = R[:, k]
		R[:, k] = R[:, jmax]
		R[:, jmax] = tmp
		
		# triangularizing
		
		v = zeros(m - k + 1)	# a Householder reflector
		v[1] = sqrt(sqnorms[k])*(R[k, k] >= 0 ? 1 : -1)
		v += R[k:m, k]
		v /= norm(v)
		
		R[k:m, :] -= 2*v*(v'*R[k:m, :])
		Q[:, k:m] -= 2*(Q[:, k:m]*v)*v'
		R[k + 1:m, k] = zeros(m - k)
		
		# maintaining a nonnegative diagonal on R
		
		s = (R[k, k] >= 0) ? 1 : -1
		R[k, k:n] *= s
		Q[:, k] *= s
		
		# updating column norms
		
		sqnorm_max = tol
		jmax = 0
		
		for j = k + 1:n
			sqnorms[j] = sqnorms[j] - R[k, j]^2
			sqnorms[j] = max(sqnorms[j], 0.)
			
			if(sqnorms[j] > sqnorm_max)
				sqnorm_max = sqnorms[j]
				jmax = j
			end
		end
		
		k += 1
	end
	
	if(sqnorm_max > tol)
		# final update to maintain a nonnegative diagonal on R
		s = (R[k, k] >= 0) ? 1 : -1
		R[k, k:n] *= s
		Q[:, k] *= s
		
		if(!silent)
			# display the final state of the factorization
			run(Cmd(["clear"]))
			println("permutation | squared column norms")
			println("----------------------------------")
			
			for j = 1:n
				@printf("     %2d     |      %8.2e       \n", perm[j], sqnorms[j])
			end
			
			println("\nQ = ")
			display(Q)
			println("\nR = ")
			display(R)
			println("\n press enter to continue...")
			
			readline()
		end
	end
	
	return QRCP(Q, R, perm)
end
