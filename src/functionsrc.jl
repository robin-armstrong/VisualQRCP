using LinearAlgebra
using Printf
using JLD2

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
	visualQRCP(A ; silent = false, showNorms = false, save = false, dest = "qrcp_hist.jld2")
	
Compute a Golub-Businger style QRCP of `A`, showing all intermediate factorizations unless `silent == true`. If `showNorms == true` then also display updates to the column norms. If `save == true`, then save the factorization history to `dest`.
"""
function visualQRCP(A::Matrix{F}; 
					silent::Bool = false, 
					showNorms::Bool = false, 
					save::Bool = false,
					dest::String = "qrcp_hist.jld2") where {F <: Real}
	m, n       = size(A)
	d          = min(m, n)
	q          = (m < n) ? 0 : 1
	perm       = Array{Int64}(range(1, n, n))
	Q          = Matrix{Float64}(I(m))
	R          = Matrix{Float64}(A)
	sqnorms    = Vector{Float64}(undef, n)
	sqnorm_max = 0.
	jmax       = 0

	if(save)
		Qhist = zeros(d+q, m, m)
		Rhist = zeros(d+q, m, n)
		Phist = zeros(Int64, d+q, n)
	end
	
	for j = 1:n
		sqnorms[j] = norm(R[:, j])^2

		if sqnorms[j] > sqnorm_max
			sqnorm_max = sqnorms[j]
			jmax       = j
		end
	end
	
	k = 1
	
	while k < d+q
		if !silent
			# display the current state of the factorization
			run(Cmd(["clear"]))
			
			if showNorms
				println("permutation | squared column norms")
				println("----------------------------------")
				
				for j = 1:n
					@printf("     %2d     |      %8.2e       \n", perm[j], sqnorms[j])
				end
			else
				println("permutation:")
				
				for j = 1:n - 1
					print(perm[j],", ")
				end
				
				println(perm[n])
			end
			
			println("\nQ factor:")
			display(Q)
			println("\nR factor:")
			display(R)
			println("\npress enter to continue...")
			
			readline()
		end
		
		if save
			Qhist[k, :, :] = Q
			Rhist[k, :, :] = R
			Phist[k, :] = perm
		end

		# swapping columns
		
		tmp        = perm[k]
		perm[k]    = perm[jmax]
		perm[jmax] = tmp
		
		tmp           = sqnorms[k]
		sqnorms[k]    = sqnorms[jmax]
		sqnorms[jmax] = tmp
		
		tmp        = R[:, k]
		R[:, k]    = R[:, jmax]
		R[:, jmax] = tmp
		
		# triangularizing
		
		v    = zeros(m - k + 1)	# a Householder reflector
		v[1] = sqrt(sqnorms[k])*(R[k, k] >= 0 ? 1 : -1)
		v  .+= R[k:m, k]
		v  ./= norm(v)
		
		R[k:m, :]    -= 2*v*(v'*R[k:m, :])
		Q[:, k:m]    -= 2*(Q[:, k:m]*v)*v'
		R[k + 1:m, k] = zeros(m - k)
		
		# maintaining a nonnegative diagonal on R
		
		s          = (R[k, k] >= 0) ? 1 : -1
		R[k, k:n] *= s
		Q[:, k]   *= s
		
		# updating column norms
		
		sqnorm_max = 0.
		jmax = 0
		
		for j = k + 1:n
			sqnorms[j] = sqnorms[j] - R[k, j]^2
			sqnorms[j] = max(sqnorms[j], 0.)
			
			if sqnorms[j] > sqnorm_max
				sqnorm_max = sqnorms[j]
				jmax       = j
			end
		end
		
		k += 1
	end
	
	# final update to maintain a nonnegative diagonal on R
	
	if k <= min(m, n)
		s          = (R[k, k] >= 0) ? 1 : -1
		R[k, k:n] *= s
		Q[:, k]   *= s
	end
	
	if !silent
		# display the final state of the factorization
		run(Cmd(["clear"]))
		
		if showNorms
			println("permutation | squared column norms")
			println("----------------------------------")
			
			for j = 1:n
				@printf("     %2d     |      %8.2e       \n", perm[j], sqnorms[j])
			end
		else
			println("permutation:")
			
			for j = 1:n - 1
				print(perm[j],", ")
			end
			
			println(perm[n])
		end
		
		println("\nQ factor:")
		display(Q)
		println("\nR factor:")
		display(R)
		println("\npress enter to continue...")
		
		readline()
	end

	if save
		Qhist[k, :, :] = Q
		Rhist[k, :, :] = R
		Phist[k, :] = perm

		@save dest Qhist Rhist Phist
	end
	
	return QRCP(Q, R, perm)
end
