requirement
- cuda>=12.1
- pytorch>=2.0.1

把预训练模型放在checkpoints路径下。
输入指令token，token之间用空格隔开，指令之间用<eos>隔开，最大输入长度为512个token，输出为每个token的embedding。预训练没有加cls token，如果要获得整段指令的embedding，可以使用平均池化。
指令预处理逻辑在Normalize.py中，目前是直接处理反汇编文件，还没来得及做修改
