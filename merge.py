import pandas as pd
import os

def merge_csv_files(file1, file2, file3, output_file):
    """
    合并三个具有相同表头的CSV文件
    
    参数:
    file1, file2, file3: 要合并的CSV文件路径
    output_file: 合并后的输出文件路径
    """
    try:
        # 读取三个CSV文件
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        df3 = pd.read_csv(file3)
        
        print(f"文件1记录数: {len(df1)}")
        print(f"文件2记录数: {len(df2)}")
        print(f"文件3记录数: {len(df3)}")
        
        # 合并数据框
        merged_df = pd.concat([df1, df2, df3], ignore_index=True)
        
        # 保存合并后的文件
        merged_df.to_csv(output_file, index=False)
        
        print(f"合并完成！总记录数: {len(merged_df)}")
        print(f"合并文件已保存至: {output_file}")
        
        return merged_df
        
    except FileNotFoundError as e:
        print(f"错误：找不到文件 - {e}")
        return None
    except Exception as e:
        print(f"合并过程中出现错误: {e}")
        return None

# 使用示例
if __name__ == "__main__":
    # 替换为你的实际文件路径
    # file1 = "submission_pre.csv"
    file1 = "generated/submission_pre.csv"
    file2 = "generated/submission_semi.csv"
    file3 = "generated/submission_final.csv"
    output_file = "generated/submission_咕噜咕噜冒泡泡.csv"
    
    result = merge_csv_files(file1, file2, file3, output_file)