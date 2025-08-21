import json
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

class TransposeAugmenter:
    def __init__(self, vocab_file: str, transpose_range: Tuple[int, int] = (-5, 6)):
        """
        初始化移调增强器
        
        Args:
            vocab_file: 词表文件路径
            transpose_range: 移调范围，默认(-5, 6)表示下移5个半音到上移6个半音
        """
        self.transpose_range = transpose_range
        self.transpose_values = list(range(transpose_range[0], transpose_range[1] + 1))
        self.num_augmentations = len(self.transpose_values)
        
        # 加载词表
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        
        # 构建反向映射：token -> (name, value)
        self.token_to_info = {}
        for token, idx in self.vocab.items():
            if token.startswith('Pitch_') and not token.startswith('PitchDrum_'):
                # 普通音高token
                pitch_value = int(token.split('_')[1])
                self.token_to_info[idx] = ('pitch', pitch_value)
            elif token.startswith('PitchDrum_'):
                # 鼓的音高token
                pitch_value = int(token.split('_')[1])
                self.token_to_info[idx] = ('drum_pitch', pitch_value)
            elif token.startswith('Chord_'):
                # 和弦token
                chord_type = token.split('_')[1]
                self.token_to_info[idx] = ('chord', chord_type)
            else:
                # 其他token（Bar, Position, Duration, Velocity, Program等）
                self.token_to_info[idx] = ('other', None)
        
        # 音域范围
        self.min_pitch = 21
        self.max_pitch = 109
        self.mid_pitch = (self.min_pitch + self.max_pitch) // 2
        
        print(f"词表加载完成，音域范围: {self.min_pitch}-{self.max_pitch}")
        print(f"中间音域: {self.mid_pitch}")
        print(f"移调范围: {transpose_range[0]} 到 {transpose_range[1]} 半音")
        print(f"增强倍数: {self.num_augmentations}x")
    
    def find_optimal_transpose(self, tokens: List[int]) -> int:
        """
        找到将曲子平移到中间音域的最优移调量
        
        Args:
            tokens: token序列
            
        Returns:
            最优移调量（半音数）
        """
        pitches = []
        for token in tokens:
            if token in self.token_to_info:
                token_type, value = self.token_to_info[token]
                if token_type == 'pitch':
                    pitches.append(value)
        
        if not pitches:
            return 0
        
        # 计算当前音域的中心
        current_min = min(pitches)
        current_max = max(pitches)
        current_center = (current_min + current_max) // 2
        
        # 计算需要移调多少才能到达中间音域
        optimal_transpose = self.mid_pitch - current_center
        
        # 确保移调后不会超出音域范围
        if current_min + optimal_transpose < self.min_pitch:
            optimal_transpose = self.min_pitch - current_min
        elif current_max + optimal_transpose > self.max_pitch:
            optimal_transpose = self.max_pitch - current_max
        
        print(f"当前音域: {current_min}-{current_max}, 中心: {current_center}")
        print(f"最优移调: {optimal_transpose} 半音")
        
        return optimal_transpose
    
    def transpose_tokens(self, tokens: List[int], transpose_steps: int) -> Optional[List[int]]:
        """
        对token序列进行移调
        
        Args:
            tokens: 原始token序列
            transpose_steps: 移调步数（半音数）
            
        Returns:
            移调后的token序列，如果移调失败返回None
        """
        if transpose_steps == 0:
            return tokens.copy()
        
        result = []
        
        for token in tokens:
            if token in self.token_to_info:
                token_type, value = self.token_to_info[token]
                
                if token_type == 'pitch':
                    # 普通音高token
                    new_pitch = value + transpose_steps
                    
                    # 检查是否超出音域
                    if new_pitch < self.min_pitch or new_pitch > self.max_pitch:
                        return None  # 移调失败
                    
                    # 构建新的token
                    new_token_name = f"Pitch_{new_pitch}"
                    if new_token_name in self.vocab:
                        result.append(self.vocab[new_token_name])
                    else:
                        return None  # 新token不在词表中
                        
                elif token_type == 'drum_pitch':
                    # 鼓的音高token不移调
                    result.append(token)
                    
                elif token_type == 'chord':
                    # 和弦token不移调
                    result.append(token)
                    
                else:
                    # 其他token（Bar, Position, Duration, Velocity, Program等）不移调
                    result.append(token)
            else:
                # 未知token，保持原样
                result.append(token)
        
        return result
    
    def augment_sequence(self, tokens: List[int]) -> List[Tuple[int, List[int]]]:
        """
        对单个序列进行移调增强
        
        Args:
            tokens: 原始token序列
            
        Returns:
            列表，每个元素为(移调量, 移调后的token序列)
        """
        # 首先找到最优移调量
        optimal_transpose = self.find_optimal_transpose(tokens)
        
        results = []
        
        # 对每个移调值进行增强
        for transpose_steps in self.transpose_values:
            # 计算实际移调量（基础移调 + 最优移调）
            actual_transpose = optimal_transpose + transpose_steps
            
            # 执行移调
            transposed_tokens = self.transpose_tokens(tokens, actual_transpose)
            
            if transposed_tokens is not None:
                results.append((actual_transpose, transposed_tokens))
            else:
                print(f"警告: 移调 {actual_transpose} 半音失败，跳过")
        
        print(f"成功生成 {len(results)} 个增强版本")
        return results
    
    def process_jsonl_file(self, input_file: str, output_dir: str, max_samples: Optional[int] = None):
        """
        处理JSONL文件，生成移调增强数据
        
        Args:
            input_file: 输入JSONL文件路径
            output_dir: 输出目录
            max_samples: 最大处理样本数（用于测试）
        """
        input_path = Path(input_file)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 为每个移调值创建输出文件
        output_files = {}
        for transpose_steps in self.transpose_values:
            filename = f"transpose_{transpose_steps:+d}.jsonl"
            output_files[transpose_steps] = open(output_path / filename, 'w', encoding='utf-8')
        
        # 统计信息
        total_processed = 0
        total_augmented = 0
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if max_samples and total_processed >= max_samples:
                    break
                
                try:
                    data = json.loads(line.strip())
                    
                    # 获取tokens字段
                    if 'tokens' in data:
                        tokens = data['tokens']
                    elif 'ids' in data:
                        tokens = data['ids']
                    else:
                        print(f"警告: 第{line_num}行没有找到tokens或ids字段，跳过")
                        continue
                    
                    # 进行移调增强
                    augmented_results = self.augment_sequence(tokens)
                    
                    # 写入输出文件
                    for actual_transpose, transposed_tokens in augmented_results:
                        # 找到对应的输出文件
                        base_transpose = actual_transpose - self.find_optimal_transpose(tokens)
                        
                        if base_transpose in output_files:
                            new_data = data.copy()
                            new_data['tokens'] = transposed_tokens
                            new_data['augment'] = {
                                'transpose': actual_transpose,
                                'base_transpose': base_transpose,
                                'optimal_transpose': self.find_optimal_transpose(tokens)
                            }
                            
                            output_files[base_transpose].write(json.dumps(new_data, ensure_ascii=False) + '\n')
                            total_augmented += 1
                    
                    total_processed += 1
                    
                    if total_processed % 100 == 0:
                        print(f"已处理 {total_processed} 个样本，生成 {total_augmented} 个增强版本")
                
                except json.JSONDecodeError as e:
                    print(f"错误: 第{line_num}行JSON解析失败: {e}")
                    continue
                except Exception as e:
                    print(f"错误: 处理第{line_num}行时发生异常: {e}")
                    continue
        
        # 关闭输出文件
        for f in output_files.values():
            f.close()
        
        print(f"\n处理完成!")
        print(f"总处理样本数: {total_processed}")
        print(f"总生成增强版本: {total_augmented}")
        print(f"平均增强倍数: {total_augmented / total_processed:.2f}x")
        
        # 输出文件统计
        for transpose_steps in self.transpose_values:
            filename = f"transpose_{transpose_steps:+d}.jsonl"
            file_path = output_path / filename
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    line_count = sum(1 for _ in f)
                print(f"{filename}: {line_count} 行")

def main():
    """主函数"""
    if len(sys.argv) < 4:
        print("用法: python transpose_augment.py <词表文件> <输入JSONL> <输出目录> [最大样本数]")
        print("示例: python transpose_augment.py tokenizer_vocab.json input.jsonl output/ 1000")
        sys.exit(1)
    
    vocab_file = sys.argv[1]
    input_file = sys.argv[2]
    output_dir = sys.argv[3]
    max_samples = int(sys.argv[4]) if len(sys.argv) > 4 else None
    
    # 检查输入文件
    if not Path(vocab_file).exists():
        print(f"错误: 词表文件不存在: {vocab_file}")
        sys.exit(1)
    
    if not Path(input_file).exists():
        print(f"错误: 输入文件不存在: {input_file}")
        sys.exit(1)
    
    # 创建增强器
    augmenter = TransposeAugmenter(vocab_file)
    
    # 处理文件
    augmenter.process_jsonl_file(input_file, output_dir, max_samples)

if __name__ == "__main__":
    main()
