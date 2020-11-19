from pyecharts.charts import Bar
from pyecharts import options as opts
from pyecharts.globals import ThemeType

bar = Bar(init_opts=opts.InitOpts(width='1200px',height='700px'))
bar.add_xaxis(["CNN_dropout",'CNN_without_dropout','lenet5_dropout','lenet5_without_dropout','random1_cifar100','random2_cifar100','ResNet_v1','ResNet_v2'])
bar.set_global_opts(title_opts=opts.TitleOpts(title="Cifar10数据扩增评估"),xaxis_opts=opts.AxisOpts(name_rotate=60,axislabel_opts={"rotate":45}))


def add(method, scores):
    bar.add_yaxis(method,scores)


def save():
    bar.render()