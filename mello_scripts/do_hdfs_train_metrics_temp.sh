
echo "#=========================== print cmd ===========================#"
train_metric=`echo "$@" | awk -F'+' '{print $1}'`  # cmd1 train metric
echo ${train_metric}
train_data=`echo "$@" | awk -F'+' '{print $2}'`    # cmd2 train data
echo ${train_data}
valid_data=`echo "$@" | awk -F'+' '{print $3}'`    # cmd3 valid data
echo ${valid_data}
ckpt_name=`echo "$@" | awk -F'+' '{print $4}'`     # cmd4 训练结果命名
save_hdfs=`echo "$@" | awk -F'+' '{print $5}'`     # cmd5 结果保存到我的hdfs路径
echo ${save_hdfs}


# 复制需要训练的指标代码，构造环境
# 训练验证数据集，从hdfs复制到本地，然后改名字
# 
my_hdfs_prefix=/home/byte_arnold_lq_mlnlc/user/yanyiming.mello

hadoop fs -get $my_hdfs_prefix/code/COMET_mello .
cd COMET_mello

pip3 install -r requirements.txt

hdfs dfs -get $my_hdfs_prefix/$train_data data/
mv data/${train_data##*/} data/train.csv
hdfs dfs -get $my_hdfs_prefix/$valid_data data/
mv data/${valid_data##*/} data/valid.csv

mkdir -p log
python3 comet/cli/train.py --cfg configs/configs_for_step_finetune/wmt20-comet-da.yaml
