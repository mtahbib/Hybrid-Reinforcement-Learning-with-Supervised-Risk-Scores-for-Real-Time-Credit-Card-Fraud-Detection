# RL Architecture and Training
import time, json, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
from pathlib import Path
from xgboost import Booster, DMatrix

from .data_loader import chrono_split
from .features import add_derived
from .rl_env import CreditFraudEnv

MODELS = Path("models"); MODELS.mkdir(exist_ok=True, parents=True)

# ---------------- Dueling Double DQN ----------------
class DuelingQ(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        h=256
        # backbone network: shared layers
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, 128), nn.ReLU(),
        )
        # value stream to see how good is the state
        self.val = nn.Sequential(nn.Linear(128,128), nn.ReLU(), nn.Linear(128,1))
        # advantage stream to see how good is each action
        self.adv = nn.Sequential(nn.Linear(128,128), nn.ReLU(), nn.Linear(128,n_actions))
    def forward(self, x):
        h = self.backbone(x)
        v, a = self.val(h), self.adv(h)
        # combine value + advantage into Q-values
        return v + (a - a.mean(dim=1, keepdim=True))

# replay buffer with prioritized sampling
class PERBuffer:
    def __init__(self, size=300_000, alpha=0.5, beta_start=0.4, beta_end=1.0, beta_steps=400_000):
        self.size=size; self.alpha=alpha
        self.beta_start=beta_start; self.beta_end=beta_end; self.beta_steps=beta_steps
        self.ptr=0; self.full=False; self.t=0
        self.obs=None; self.nobs=None
        self.act=None; self.rew=None; self.done=None
        self.prior = None
    def _ensure(self, obs_shape):
        if self.obs is None:
            self.obs   = np.zeros((self.size, *obs_shape), dtype=np.float32)
            self.nobs  = np.zeros((self.size, *obs_shape), dtype=np.float32)
            self.act   = np.zeros((self.size,), dtype=np.int64)
            self.rew   = np.zeros((self.size,), dtype=np.float32)
            self.done  = np.zeros((self.size,), dtype=np.float32)
            self.prior = np.ones((self.size,), dtype=np.float32)
    def add(self, o,a,r,d,o2):
        # store transition
        self._ensure(o.shape)
        i=self.ptr
        self.obs[i]=o; self.act[i]=a; self.rew[i]=r
        self.done[i]=d; self.nobs[i]=o2
        self.prior[i]=max(1.0, self.prior.max())  # keep >=1
        self.ptr=(self.ptr+1)%self.size
        self.full = self.full or self.ptr==0
        self.t += 1
    def _beta(self):
        if self.beta_steps<=0: return self.beta_end
        frac = min(1.0, self.t / self.beta_steps)
        return self.beta_start + frac*(self.beta_end - self.beta_start)
    def sample(self, batch_size=256):
        # sample with probability proportional to priority
        N = self.size if self.full else self.ptr
        prob = self.prior[:N] ** self.alpha
        prob /= prob.sum()
        idx = np.random.choice(N, size=min(batch_size, N), p=prob)
        beta = self._beta()
        w = (N * prob[idx]) ** (-beta)
        w /= w.max()
        return (self.obs[idx], self.act[idx], self.rew[idx], self.done[idx], self.nobs[idx], idx, w.astype(np.float32))
    def update_prior(self, idx, td):
        # update priorities with TD error
        self.prior[idx] = np.clip(np.abs(td), 0.0, 5.0) + 1e-6

# soft update target network
def soft_update(target, online, tau=0.005):
    with torch.no_grad():
        for tp, p in zip(target.parameters(), online.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * p.data)

# helper: get xgboost scores for RL state input
def xgb_scores_for(Xtr, Xva, Xte):
    booster = Booster()
    booster.load_model("models/xgb_base.json")
    dtr, dva, dte = DMatrix(Xtr), DMatrix(Xva), DMatrix(Xte)
    return (booster.predict(dtr).astype(np.float32),
            booster.predict(dva).astype(np.float32),
            booster.predict(dte).astype(np.float32))

# rollout once and compute cost
@torch.no_grad()
def rollout_cost(env, q, steps=None, device='cpu'):
    steps = steps or env.episode_len
    obs,_ = env.reset()
    actions=[]
    for _ in range(steps):
        t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        a = int(q(t).argmax(dim=1).item())
        actions.append(a)
        obs, r, done, _, _ = env.step(a)
        if done: break
    # calculate total cost per 1k transactions
    y_true = env.y[env.idx]
    
    rc = dict(tp_block=5.0, fp_block=-2.0, fn=-10.0, tn=0.2, review_cost=-0.2, review_catch=3.0, over_budget=-5.0) #Rewards 
    cost=0.0 #actions 
    for a, label in zip(actions, y_true):
        if a==2:   cost += rc['tp_block'] if label==1 else rc['fp_block']
        elif a==0: cost += rc['tn'] if label==0 else rc['fn']
        else:      cost += rc['review_cost'] + (rc['review_catch'] if label==1 else 0.0)
    return 1000.0 * cost / len(y_true)

# main training loop
def train_dqn(steps=400_000, batch_size=256, review_rate=0.05, episode_len=8000, seed=42):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load dataset and split
    df = pd.read_csv("data/creditcard.csv")
    sp = chrono_split(df)
    Xtr = add_derived(sp.X_train); Xva = add_derived(sp.X_val); Xte = add_derived(sp.X_test)
    s_tr, s_va, s_te = xgb_scores_for(Xtr, Xva, Xte)

    # create RL environments
    env_tr = CreditFraudEnv(Xtr, sp.y_train, s_tr, review_rate=review_rate, episode_len=episode_len, seed=seed,   strict_budget=True)
    env_va = CreditFraudEnv(Xva, sp.y_val,   s_va, review_rate=review_rate, episode_len=min(4000, len(Xva)), seed=seed+1, strict_budget=True)

    obs_dim = env_tr.observation_space.shape[0]
    n_actions = env_tr.action_space.n

    # Q-networks (online + target)
    q = DuelingQ(obs_dim, n_actions).to(device)
    tgt = DuelingQ(obs_dim, n_actions).to(device)
    tgt.load_state_dict(q.state_dict())

    opt = torch.optim.AdamW(q.parameters(), lr=1e-4, weight_decay=1e-4)

    per = PERBuffer(size=300_000, alpha=0.5, beta_start=0.4, beta_end=1.0, beta_steps=steps)

    eps_start, eps_end, eps_decay = 0.20, 0.05, steps  # epsilon-greedy
    gamma = 0.98
    min_buf = 10_000
    target_clip = 25.0

    o,_ = env_tr.reset()
    best_va_cost = float('inf')
    save_every = 50_000
    t0=time.time(); losses=[]

    for t in range(1, steps+1):
        # epsilon-greedy exploration
        eps = max(eps_end, eps_start - (eps_start-eps_end)*(t/eps_decay))
        if np.random.rand() < eps:
            a = np.random.randint(n_actions)
        else:
            with torch.no_grad():
                a = int(q(torch.as_tensor(o, dtype=torch.float32, device=device).unsqueeze(0)).argmax(dim=1).item())

        # step environment
        o2, r, d, _, _ = env_tr.step(a)
        per.add(o, a, r, float(d), o2)
        o = o2 if not d else env_tr.reset()[0]

        # learn after buffer has enough samples
        if (per.ptr > min_buf or per.full):
            obs, act, rew, done, nobs, idx, iw = per.sample(batch_size)
            dvc = device
            obs  = torch.as_tensor(obs,  dtype=torch.float32, device=dvc)
            nobs = torch.as_tensor(nobs, dtype=torch.float32, device=dvc)
            act  = torch.as_tensor(act,  dtype=torch.long,   device=dvc)
            rew  = torch.as_tensor(rew,  dtype=torch.float32, device=dvc)
            done = torch.as_tensor(done, dtype=torch.float32, device=dvc)
            iw   = torch.as_tensor(iw,   dtype=torch.float32, device=dvc)

            # Double DQN target
            q_sa = q(obs).gather(1, act.view(-1,1)).squeeze(1)
            with torch.no_grad():
                na = q(nobs).argmax(dim=1, keepdim=True)
                tgt_q = tgt(nobs).gather(1, na).squeeze(1)
                y = rew + (1.0 - done) * gamma * tgt_q
                y = torch.clamp(y, -target_clip, target_clip)

            td = y - q_sa
            loss_vec = F.smooth_l1_loss(q_sa, y, reduction='none')
            loss = (iw * loss_vec).mean()

            if not torch.isnan(loss):
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(q.parameters(), 5.0)
                opt.step()
                soft_update(tgt, q, tau=0.005)

                per.update_prior(idx, td.detach().abs().cpu().numpy())
                losses.append(float(loss.item()))

        if t % 20_000 == 0:
            avg_loss = float(np.mean(losses[-2000:])) if losses else float('nan')
            elapsed = (time.time()-t0)/60.0
            print(f"[{t}/{steps}] avg_loss={avg_loss:.5f} elapsed={elapsed:.1f}m eps={eps:.3f}")

        # save best model by validation cost
        if t % save_every == 0:
            q.eval()
            va_cost = rollout_cost(env_va, q, device=device)
            print(f"  -> val cost/1k = {va_cost:.3f} at t={t}")
            if va_cost < best_va_cost:
                best_va_cost = va_cost
                torch.save(q.state_dict(), MODELS/'dqn_policy.pt')
                with torch.no_grad():
                    dummy = torch.randn(1, obs_dim, device=device)
                    ts = torch.jit.trace(q, dummy)
                    ts.save(str(MODELS/'dqn_policy.ts'))
                print(f"  ** saved new best (val cost/1k {best_va_cost:.3f})")

    # final save
    torch.save(q.state_dict(), MODELS/'dqn_policy_last.pt')
    with torch.no_grad():
        dummy = torch.randn(1, obs_dim, device=device)
        ts = torch.jit.trace(q.eval(), dummy)
        ts.save(str(MODELS/'dqn_policy_last.ts'))
    print("Saved:", MODELS/'dqn_policy_last.pt', "and", MODELS/'dqn_policy_last.ts')

def main():
    train_dqn(
        steps=400_000,
        batch_size=256,
        review_rate=0.05,
        episode_len=8000,
        seed=42
    )

if __name__ == "__main__":
    main()
