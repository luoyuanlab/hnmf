function [F, Gp, Gg, errorp, kl] = hnmf(Xp, Xg, k, lambda, timelimit, maxiter, tol, maxsubiter, seed)
  %% Hybrid NMF by alternating projected gradients
  %% Author: Yuan Luo, yuan.hypnos.luo@gmail.com
  %% n - number of patients
  %% mp - number of phenotypes
  %% mg - number of genotypes
  %% k - number of patient groups
  %% Xp - patient by phenotype matrix, continuous value (n x mp)
  %% Xg - patient by genotype matrix, count value (n x mg)
  %% F - patient group assignment matrix (k x n)
  %% Gp - phenotype group assignment matrix (k x mp)
  %% Gg - genotype group assignment matrix (k x mg)
  %% initF,initGp,initGg - initial solution
  %% tol - tolerance for a relative stopping condition
  %% timelimit, maxiter - limit of time and iterations
  if nargin == 8,
    seed = 20170829;
  end
  rng(seed,'twister');
  errorp=[];
  kl=[];
  toleps = 1e-6;
  [np, mp] = size(Xp);
  [ng, mg] = size(Xg);
  assert(np==ng, sprintf('np (%d) != ng (%d)', np, ng))
  n = np;
  Eg = ones(ng, mg); Ef = ones(mg,n);
  F = max(0, rand(k, n));
  Gp = max(0, rand(k, mp));
  Gg = max(0, rand(k, mg));

  initt = cputime;
  gradGp = (F*F')*Gp - F*Xp;
  Xghat = F'*Gg + toleps;
  Xgtilde = Xg ./ Xghat;
  gradGg = F*(Eg - Xgtilde);
  gradF = lambda*(-Gp*Xp' + Gp*Gp'*F) + Gg*(Ef-Xgtilde');

  initgrad =  norm(gradF, 'fro') + norm(gradGp, 'fro') + norm(gradGg, 'fro');
  fprintf('Init gradient norm %f\n', initgrad);

  tolGp = tol * initgrad; 
  tolGg = tol * initgrad; 
  tolF = tol * initgrad; 
  ll = Xg.*log(Xg+toleps) - Xg;
  KLbase = sum(sum(ll));
  fprintf('KLbase=%.2f\n', KLbase);
  for iter=1:maxiter,
    %% stopping condition
    nGradF = norm(gradF(gradF<0 | F>0), 'fro');
    nGradGp = norm(gradGp(gradGp<0 | Gp>0), 'fro');
    nGradGg = norm(gradGg(gradGg<0 | Gg>0), 'fro');

    projnorm = nGradF + nGradGp + nGradGg;
    errorpi = norm(Xp-F'*Gp,'fro');
    errorp=[errorp;errorpi];
    Xghat = F'*Gg + toleps; Xphat = F'*Gp;
    fprintf('max(Xphat)=%.2f, max(Xghat)=%.2f\n', max(max(Xphat)), max(max(Xghat)));
    nll = Xghat - Xg.*log(Xghat);
    KLloss=sum(sum(nll)) + KLbase;
    kl=[kl;KLloss];
    fprintf('iter=%d, pheno-fro=%.2f, geno-KL=%.2f\n', iter, errorpi, KLloss);

    if projnorm <= tol*initgrad+toleps || cputime-initt > timelimit,
      fprintf('main break\n')
      break;
    end
    [F,gradF,iterF,suff_decrF] = subF(Xp,Xg,Gp,Gg,F,lambda,tolF,maxsubiter,toleps);

    if iterF==1,
      tolF = 0.1 * tolF;
    end
    Xghat = F'*Gg+toleps;
    nll = Xghat - Xg.*log(Xghat);
    KLloss=sum(sum(nll)) + KLbase;
    errorpi = norm(Xp-F'*Gp,'fro');
    errorp=[errorp;errorpi]; kl=[kl;KLloss];

    [Gp,gradGp,iterGp,suff_decrGp] = subGp(Xp,F,Gp,tolGp,maxsubiter,toleps);
    if iterGp==1,
      tolGp = 0.1 * tolGp;
    end
    Xghat = F'*Gg+toleps;
    nll = Xghat - Xg.*log(Xghat);
    KLloss=sum(sum(nll)) + KLbase;
    errorpi = norm(Xp-F'*Gp,'fro');
    errorp=[errorp;errorpi]; kl=[kl;KLloss];

    [Gg,gradGg,iterGg,suff_decrGg] = subGg(Xg,F,Gg,tolGg,maxsubiter,toleps);
    if iterGg==1,
      tolGg = 0.1 * tolGg;
    end
    Xghat = F'*Gg+toleps;
    nll = Xghat - Xg.*log(Xghat);
    KLloss=sum(sum(nll)) + KLbase;
    errorpi = norm(Xp-F'*Gp,'fro');
    errorp=[errorp;errorpi]; kl=[kl;KLloss];
    if ~suff_decrF & ~suff_decrGp & ~suff_decrGg,
      fprintf('alt break, suff_decrF=%d, suff_decrGp=%d, suff_decrGg=%d\n', suff_decrF, suff_decrGp, suff_decrGg);
      break;
    end
  end
  fprintf('\nIter = %d Final proj-grad norm %f\n', iter, projnorm);
end

function [Gp,grad,iter,suff_decr_ever] = subGp(Xp,F,initGp,tol,maxiter,toleps)
  %% with F fixed, square error is quadratic in Gp
  %% Gp, grad: output solution and gradient
  %% iter: #iterations used
  %% Xp, F: constant matrices
  %% initGp: initial solution
  %% tol: stopping tolerance
  %% maxiter: limit of iterations
  Gp = initGp; hessian = F*F'; FXp = F*Xp;

  alpha = 1; beta = 0.1; sigma = 0.01; suff_decr_ever = 0;
  for iter=1:maxiter,
    grad = hessian*Gp - FXp;
    projgrad = norm(grad(grad < 0 | Gp >0));

    if projgrad <= tol+toleps,
      fprintf('Gp projgrad break at iter=%d, projgrad=%f\n', iter, projgrad);
      break
    end

    %% search step size
    for inner_iter=1:20,
      Gpn = max(Gp - alpha*grad, 0); d = Gpn-Gp;
      gradd=sum(sum(grad.*d)); dQd = sum(sum((hessian*d).*d));

      suff_decr = ((1-sigma)*gradd + 0.5*dQd) < -toleps;
      if suff_decr,
	suff_decr_ever = 1;
      end

      if inner_iter==1,
	decr_alpha = ~suff_decr; Gpp = Gp;
      end
      if decr_alpha,
	if suff_decr,
	  Gp = Gpn; break;
	else
	  alpha = alpha * beta;
	end
      else
	if ~suff_decr || all(all(Gpp == Gpn)),
	  fprintf('subGp inner break at iter=%d, suff_decr=%d, no-update=%d\n', iter, suff_decr, all(all(Gpp == Gpn)));
	  Gp = Gpp; break;
	else
	  alpha = alpha/beta; Gpp = Gpn;
	end
      end
    end
  end

  if iter==maxiter,
    fprintf('Max iter in subGp, suff_decr_ever=%d\n', suff_decr_ever);
  end
end

function [Gg,grad,iter,suff_decr_ever] = subGg(Xg,F,initGg,tol,maxiter,toleps)
  %% Gg, grad: output solution and gradient
  %% iter: #iterations used
  %% Xg, F: constant matrices
  %% initGg: initial solution
  %% tol: stopping tolerance
  %% maxiter: limit of iterations
  Gg = initGg;
  [n, mg] = size(Xg); [k, n]= size(F);
  Eg = ones(n, mg); FEg = F*Eg;
  alpha = 1; beta = 0.1; sigma = 0.01; suff_decr_ever = 0;
  for iter=1:maxiter,
    Xghat = F'*Gg + toleps;
    Xgtilde = Xg ./ Xghat;
    grad = FEg - F*Xgtilde;
    projgrad = norm(grad(grad < 0 | Gg >0),'fro');

    if projgrad <= tol + toleps,
      fprintf('Gg projgrad break at iter=%d, projgrad=%f\n', iter, projgrad);
      break
    end

    %% search step size
    for inner_iter=1:20,
      Ggn = max(Gg - alpha*grad, 0); d = Ggn-Gg;
      Xghat = F'*Gg + toleps; Xghatn = F'*Ggn + toleps;
      nlln = Xghatn - Xg.*log(Xghatn);
      nll = Xghat - Xg.*log(Xghat);
      fund = sum(sum(nlln)) - sum(sum(nll)); % function diff
      gradd = sum(sum(grad.*d));
      suff_decr = fund < (sigma*gradd - toleps);
      if suff_decr,
	suff_decr_ever = 1;
      end
      if inner_iter==1,
	decr_alpha = ~suff_decr; Ggp = Gg;
      end
      if decr_alpha,
	if suff_decr,
	  Gg = Ggn; break;
	else
	  alpha = alpha * beta;
	end
      else
	if ~suff_decr || all(all(Ggp == Ggn)),
	  fprintf('subGg inner break at iter=%d, suff_decr=%d, no-update=%d\n', iter, suff_decr, all(all(Ggp == Ggn)));

	  Gg = Ggp; break;
	else
	  alpha = alpha/beta; Ggp = Ggn;
	end
      end
    end
  end

  if iter==maxiter,
    fprintf('Max iter in subGg, suff_decr_ever=%d\n', suff_decr_ever);
  end
end

function [F,grad,iter,suff_decr_ever] = subF(Xp,Xg,Gp,Gg,initF,lambda,tol,maxiter,toleps)
  %% F, grad: output solution and gradient
  %% iter: #iterations used
  %% Gp, Gg: constant matrices
  %% initF: initial solution
  %% tol: stopping tolerance
  %% maxiter: limit of iterations
  F = initF;
  [k, n]= size(F);
  [np, mp] = size(Xp);
  [ng, mg] = size(Xg);
  Ef = ones(mg,ng);
  alpha = 1; beta = 0.1; sigma = 0.01; suff_decr_ever = 0;
  GpXpt = Gp*Xp'; GpGpt = Gp*Gp'; GgEf = Gg*Ef;
  for iter=1:maxiter,
    Xghat = F'*Gg + toleps;
    Xgtilde = Xg ./ Xghat;
    gradq = lambda*(-GpXpt + GpGpt*F); gradkl = GgEf -Gg*Xgtilde';
    grad =  gradq + gradkl;

    projgrad = norm(grad(grad < 0 | F >0), 'fro');

    if projgrad <= tol + toleps,
      fprintf('F projgrad break at iter=%d, projgrad=%f\n', iter, projgrad);
      break
    end

    %% search step size
    for inner_iter=1:20
      Fn = max(F - alpha*grad, 0);

      d = Fn-F;
      Xghat = F'*Gg + toleps; Xghatn = Fn'*Gg + toleps;
      nlln = Xghatn - Xg.*log(Xghatn);
      nll = Xghat - Xg.*log(Xghat);
      fundg = sum(sum(nlln)) - sum(sum(nll)); % function diff
      diffn = Xp - Fn'*Gp; diff = Xp - F'*Gp;

      fundp = norm(diffn, 'fro')^2 - norm(diff, 'fro')^2;
      gradd = sum(sum(grad.*d)); fund = fundg + fundp*lambda;
      suff_decr = fund < (sigma*gradd - toleps);
      if suff_decr,
	suff_decr_ever = 1;
      end
      if inner_iter==1,
	decr_alpha = ~suff_decr; Fp = F;
      end
      if decr_alpha,
	if suff_decr,
	  F = Fn; break;
	else
	  alpha = alpha * beta;
	end
      else
	if ~suff_decr || all(all(Fp == Fn)),
	  fprintf('subF inner break at iter=%d, suff_decr=%d, no-update=%d\n', iter, suff_decr, all(all(Fp == Fn)));
	  F = Fp; break;
	else
	  alpha = alpha/beta; Fp = Fn;
	end
      end
    end
  end

  if iter==maxiter,
    fprintf('Max iter in subF, suff_decr_ever=%d\n', suff_decr_ever);
  end
end
