import json
import pymongo

info = [{
	"name": "results_cosmetics_data",
	"introduction": "化妆品数据",
	"columns": [
		{"types": "类型"},
		{"cic_site": "在华责任单位地址"},
		{"approval_status": "批件状态"},
		{"product_name_remark": "产品名称备注"},
		{"cates": "产品类别"},
		{"from_company": "生产企业"},
		{"from_company_english": "生产企业（英文）"},
		{"expire_date": "批件过期时间"},
		{"reamrk": "备注"},
		{"name": "产品名"},
		{"ename": "产品名（英文）"},
		{"fce_site": "生产企业地址"},
		{"approval_date": "批准日期"},
		{"from_nation": "生产国"},
		{"company_in_china": "在华责任单位"},
		{"production_license": "围产许可证号"},
		{"approval_no": "批准文号"}
	]
}, {
	"name": "results_gsp_data",
	"introduction": "药品GSP认证数据",
	"columns": [
		{"name": "企业名称"},
		{"location": "企业所在地"},
		{"approval_date": "发证日期"},
		{"expire_date": "过期时间"},
		{"certi_no": "证书编号"},
		{"notice_no": "公告号"},
		{"auth_range": "认证范围"},
		{"operate_range": "经营范围"},
		{"address": "经营地址"}
	]
}, {
	"name": "results_juncai_new",
	"introduction": "军队招标数据",
	"columns": [
		{"region": "地址"},
		{"item_cate": "项目类型"},
		{"url": "网址"},
		{"allcontent": "公告全文"},
		{"addtime": "发文时间"},
		{"procure_style": "获取类型"},
		{"item_no": "项目编号"},
		{"title": "标题"}
	]
}, {
	"name": "results_medical_institution",
	"introduction": "医疗机构数据",
	"columns": [
		{"name": "医院名称"},
		{"outpatient": "企业所在地"},
		{"address": "地址"},
		{"route": "乘车路线"},
		{"zipcode": "邮编"},
		{"mainEquip": "主要设备"},
		{"level": "等级"},
		{"cate": "类型"},
		{"medical_insuranse": "医保定点"},
		{"introduction": "介绍"},
		{"bed_num": "病床数量"},
		{"telephone": "电话"},
		{"website": "网址"},
		{"special": "特色专科"}
	]
}, {
	"name": "results_medical",
	"introduction": "国产药品数据",
	"columns": [
		{"name": "药品名"},
		{"english_name": "药品名（英文）"},
		{"product_cate": "产品类别"},
		{"approval_no": "批准文号"},
		{"approval_date": "批准日期"},
		{"original_no": "原文号"},
		{"size_reagents": "规格剂型"},
		{"drug_code": "药品本位码"},
		{"code_remark": "本位码备注"},
		{"production_company": "生产单位"},
		{"production_addr": "生产地址"}
	]
}, {
	"name": "results_medicineEquip_data",
	"introduction": "医疗器械数据",
	"columns": [
		{"name": "名称"},
		{"productAddr": "生产场所"},
		{"address": "地址"},
		{"perfomance_compos": "性能及组成"},
		{"expiredate": "有效期"},
		{"standard": "产品标准"},
		{"approvaldate": "批准日期"},
		{"registerNo": "注册号"},
		{"zipcode": "邮编"},
		{"product": "企业名称"},
		{"size": "规格型号"},
		{"scope": "适用范围"},
		{"remark": "备注"}
	]
}, {
	"name": "results_lianjia_ershou",
	"introduction": "链家二手房数据",
	"columns": [
		{"inner_url": "内页链接"},
		{"hid": "房屋编号"},
		{"title": "标题"},
		{"position": "地址"},
		{"houseinfo": "房屋信息"},
		{"republish": "关注情况"},
		{"totalprice": "总价"},
		{"unitprice": "单价"},
		{"layout": "房屋户型"},
		{"build_area": "建筑面积"},
		{"inter_area": "套内面积"},
		{"orientation": "装修情况"},
		{"dicoration": "房屋朝向"},
		{"elevator": "配备电梯"},
		{"floor": "所在楼层"},
		{"unit": "户型结构"},
		{"build_type": "建筑类型"},
		{"structure": "建筑结构"},
		{"proportion": "梯户比例"},
		{"height": "楼层高度"},
		{"listing_time": "挂牌时间"},
		{"last_trade": "上次交易时间"},
		{"house_age": "房屋年限"},
		{"mortgage": "抵押信息"},
		{"owner": "交易权属"},
		{"purpose": "房屋用途"},
		{"property_own": "产权所属"},
		{"backup": "房本备件"},
		{"attribute": "房源特色"},
		{"house_picture": "房屋图片"},
		{"perhouseinfo": "户型分间"},
		{"province": "省份"},
		{"city": "城市"}
	]
}]

# 存储
myclient = pymongo.MongoClient("mongodb://admin:123456@10.0.1.40/?authSource=admin")
mydb = myclient["admin"]
mycol = mydb["collection_introduction"]
mycol.insert_one(info)
# for i in info:
# 	mycol.insert_one(i)

# 查询
last_one = mycol.find_one()
pipeline = []
result = mycol.aggregate(pipeline)
for i in result:
	print(i)

