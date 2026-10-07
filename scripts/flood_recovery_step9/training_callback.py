from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

# 训练期间记录 Episode 结果和保存 checkpoint
# 记录 completed episodes；
# recovered；
# recovery time；
# 当前训练 timestep。
# 定期保存 ppo_checkpoint_<timesteps>.zip
# training_episode_metrics.json
class RecoveryTrainingCallback(BaseCallback):
    def __init__(self, output_dir:Path, checkpoint_every_steps:int=25000, verbose:int=1):
        super().__init__(verbose)
        self.output_dir=Path(output_dir); self.output_dir.mkdir(parents=True,exist_ok=True)
        self.checkpoint_every_steps=int(checkpoint_every_steps)
        self.completed=0; self.recovered=0; self.recovery_times=[]; self.records=[]
    def _on_step(self)->bool:
        for done,info in zip(self.locals.get('dones',[]), self.locals.get('infos',[])):
            if not done: continue
            self.completed+=1
            ok=bool(info.get('recovered',False)); self.recovered+=int(ok)
            rt=info.get('recovery_time_s')
            if rt is not None: self.recovery_times.append(float(rt))
            self.records.append({'num_timesteps':int(self.num_timesteps),'recovered':ok,'recovery_time_s':rt,'co2_total_kg':info.get('co2_total_kg'),'hard_braking_events_total':info.get('hard_braking_events_total'),'queue_exposure_veh_h':info.get('queue_exposure_veh_h'),'total_time_loss_veh_h':info.get('total_time_loss_veh_h')})
        if self.completed:
            self.logger.record('recovery/completed_episodes',self.completed)
            self.logger.record('recovery/recovery_rate',self.recovered/self.completed)
            if self.recovery_times: self.logger.record('recovery/mean_recovery_time_s',float(np.mean(self.recovery_times)))
        if self.checkpoint_every_steps>0 and self.num_timesteps % self.checkpoint_every_steps==0:
            self.model.save(self.output_dir/f'ppo_checkpoint_{self.num_timesteps}')
            self._write()
        return True
    def _on_training_end(self): self._write()
    def _write(self):
        payload={'completed_episodes':self.completed,'recovered_episodes':self.recovered,'recovery_rate':self.recovered/self.completed if self.completed else None,'mean_recovery_time_s':float(np.mean(self.recovery_times)) if self.recovery_times else None,'episodes':self.records}
        (self.output_dir/'training_episode_metrics.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
