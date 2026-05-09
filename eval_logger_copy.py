import datetime
import os 
import json
import shutil


class EvalLogger:
    def __init__(self,log_root='eval_logs'):
        self.log_root=log_root
        self.run_id=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir=f'{self.log_root}/{self.run_id}'
        os.makedirs(self.run_dir,exist_ok=True)

    def log_config(self,config:dict):
        with open(f'{self.run_dir}/config.json','w',encoding='utf-8') as f:
                  json.dump(config,f,ensure_ascii=False,indent=2)  #ensure_ascii=false是为了让json中的中文正常显示，indent=2是为了让json文件每一个键值对之间换行并缩进两个空格

    def log_metrics(self,metrics:dict):
          with open(f'{self.run_dir}/metrics.json','w',encoding='utf-8') as f:
                json.dump(metrics,f,ensure_ascii=False,indent=2)

    def save_predictions(self,pred_path:str):
          if os.path.exists(pred_path):
                shutil.copy2(pred_path,f'{self.run_dir}/predictions.jsonl')

    def save_badcases(self,badcases:list):
          with open(f'{self.run_dir}/badcases.jsonl','w',encoding='utf-8') as f:
                for line in badcases:
                      # ⚠️ BUG 修复: JSONL 每行一个完整JSON对象，不能加 indent=2
                      #    indent=2 会让一个dict占3行，破坏JSONL格式
                      f.write(json.dumps(line,ensure_ascii=False)+'\n')
    

    def update_summary_csv(self,config:dict,metrics:dict):
          csv_path=f'{self.log_root}/summary.csv'
          # ⚠️ BUG 修复: prefix 传 'cfg' 而不是 'cfg_'
          #    _flatten_dict 内部会加 _，传 'cfg_' 会导致双下划线 cfg__a
          flat_config=EvalLogger._flatten_dict(config,prefix='cfg')
          flat_metrics=EvalLogger._flatten_dict(metrics)
          # ⚠️ BUG 修复: 必须用展平后的 flat_config/flat_metrics
          #    而不是原始的 config/metrics，否则 nested dict 不展开
          #    dict.keys() 返回 dict_keys 不能直接 + 列表，必须包 list()
          head=['run_id']+list(flat_config.keys())+list(flat_metrics.keys())
          row=[self.run_id]+list(flat_config.values())+list(flat_metrics.values())
          file_exists=os.path.isfile(csv_path)
          if not file_exists:
                with open(csv_path,'w',encoding='utf-8') as f:
                      f.write(','.join(head)+'\n')
                      f.write(','.join(str(i) for i in row)+'\n')
          else:
                with open(csv_path,'a',encoding='utf-8') as f:
                      f.write(','.join(str(i) for i in row)+'\n')

          print(f'summary updated:{csv_path}')

    @staticmethod
    def _flatten_dict(config:dict,prefix:str='')->dict:
          flat={}
          for key,val in config.items():
                
                flat_key=f'{prefix}_{key}' if prefix else key
                if isinstance(val,dict):
                      flat.update(EvalLogger._flatten_dict(val,prefix=flat_key))
                else:
                      flat[flat_key]=val

          return flat
    
    def print_summary(self,metrics:dict):
          
          
          print('='*50)

          for key,val in metrics.items():
                if isinstance(val,float):
                      print(f'{key:.20s}:{val:.4f}')
                else:
                        print(f'{key:.20s}:{val}')


    @classmethod
    def load_summary(cls,log_root='eval_logs')->list:
          import csv
          with open(f'{log_root}/summary.csv','r',encoding='utf-8') as f:
                reader=csv.DictReader(f)
                return list(reader)
    
    @classmethod
    def print_comparison(cls,log_root='eval_logs'):
        """打印所有 runs 的对比表"""
        rows = cls.load_summary(log_root)
        if not rows:
            return

        print("\n" + "=" * 80)
        print("📊 ALL RUNS COMPARISON")
        print("=" * 80)

        # 提取所有列名
        headers = list(rows[0].keys())
        # 格式化打印
        col_widths = {h: max(len(h), 12) for h in headers}
        for row in rows:
            for h in headers:
                col_widths[h] = max(col_widths.get(h, 12), len(str(row[h])), 12)

        # 打印表头
        header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
        print(header_line)
        print("-" * len(header_line))

        # 打印数据
        for row in rows:
            data_line = " | ".join(str(row[h]).ljust(col_widths[h]) for h in headers)
            print(data_line)

        print("=" * 80)

    
        
          
                
                                
                    


        
          
