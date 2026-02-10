# -*- coding: utf-8 -*-
"""
管理命令：初始化糖尿病视网膜病变治疗方案模板
使用方法：python manage.py init_treatment_templates
"""
from django.core.management.base import BaseCommand
from treatment.models import TreatmentPlanTemplate


class Command(BaseCommand):
    help = '初始化糖尿病视网膜病变(DR)治疗方案模板'

    def handle(self, *args, **options):
        self.stdout.write('开始创建治疗方案模板...\n')

        # 清空现有模板（可选）
        # TreatmentPlanTemplate.objects.all().delete()
        # self.stdout.write('已清空现有模板\n')

        # 创建各DR分级的治疗方案模板
        templates = [
            self._create_dr0_template(),      # 无DR
            self._create_dr1_template(),      # 轻度NPDR
            self._create_dr2_template(),      # 中度NPDR
            self._create_dr3_template(),      # 重度NPDR
            self._create_dr4_template(),      # PDR（增殖性DR）
        ]

        self.stdout.write(self.style.SUCCESS(f'\n成功创建 {len(templates)} 个治疗方案模板！'))

        # 打印模板列表
        self.stdout.write('\n模板列表：')
        for template in TreatmentPlanTemplate.objects.filter(is_active=True).order_by('dr_grade'):
            self.stdout.write(f'  - DR {template.dr_grade}级: {template}')

    def _create_dr0_template(self):
        """无DR（正常）"""
        return TreatmentPlanTemplate.objects.update_or_create(
            dr_grade=0,
            defaults={
                'dr_grade': 0,
                'priority': 100,
                'is_active': True,
                'title': '无DR - 预防与监测方案',
                # 基础管理目标
                'blood_sugar_target': {
                    'fasting': '4.4-7.0 mmol/L',
                    'postprandial': '<10.0 mmol/L',
                    'hba1c': '<7.0%'
                },
                'blood_pressure_target': {
                    'systolic': '<140 mmHg',
                    'diastolic': '<90 mmHg'
                },
                'lipid_management': 'LDL-C < 3.4 mmol/L，定期监测血脂水平',
                # 眼科治疗
                'anti_vegf_treatment': {},
                'laser_treatment': {},
                'surgical_treatment': '',
                # 药物（全身用药）
                'medications': [
                    {
                        'name': '根据内分泌科建议',
                        'category': '降糖药',
                        'dosage': '遵内分泌科医嘱',
                        'duration': '长期',
                        'notes': '定期复查血糖，调整用药'
                    }
                ],
                # 生活方式
                'diet_guidance': '''• 控制总热量摄入，维持理想体重
• 主食粗细搭配，减少精白米面
• 多吃蔬菜水果，保证膳食纤维摄入
• 限制饱和脂肪和反式脂肪摄入
• 限制添加糖摄入（<50g/天）''',
                'exercise_guidance': '''• 每周至少150分钟中等强度有氧运动
• 如快走、游泳、骑自行车
• 餐后1小时运动效果最佳
• 避免剧烈运动和憋气动作
• 运动时随身携带糖果，防止低血糖''',
                'lifestyle_advice': '''• 定期监测血糖、血压、血脂
• 每年进行一次全面眼科检查
• 保持规律作息，避免熬夜
• 戒烟限酒
• 保持心情愉悦，避免过度紧张''',
                # 随访监测
                'follow_up_plan': {
                    'interval_days': 365,
                    'check_items': ['视力检查', '眼底照相', '血糖检测', '血压测量'],
                    'notes': '一年一次全面眼科检查'
                },
                'monitoring_plan': [
                    {'item': '空腹血糖', 'frequency': '每周至少1次', 'target': '4.4-7.0 mmol/L'},
                    {'item': '餐后血糖', 'frequency': '每周至少1次', 'target': '<10.0 mmol/L'},
                    {'item': 'HbA1c', 'frequency': '每3个月', 'target': '<7.0%'},
                    {'item': '血压', 'frequency': '每周测量', 'target': '<140/90 mmHg'},
                    {'item': '血脂', 'frequency': '每6个月', 'target': 'LDL-C<3.4mmol/L'},
                ],
                'warning_symptoms': '''如出现以下症状，请立即就医：
• 视力突然下降或视物模糊
•眼前出现黑影或漂浮物
•视野出现暗区或缺损
•眼睛疼痛或发红''',
                'precautions': '无特殊治疗，定期随访即可'
            }
        )[0]

    def _create_dr1_template(self):
        """轻度NPDR"""
        return TreatmentPlanTemplate.objects.update_or_create(
            dr_grade=1,
            defaults={
                'dr_grade': 1,
                'priority': 90,
                'is_active': True,
                'title': '轻度NPDR - 强化管理方案',
                # 基础管理目标
                'blood_sugar_target': {
                    'fasting': '4.4-7.0 mmol/L',
                    'postprandial': '<10.0 mmol/L',
                    'hba1c': '<7.0%'
                },
                'blood_pressure_target': {
                    'systolic': '<130 mmHg',
                    'diastolic': '<80 mmHg'
                },
                'lipid_management': 'LDL-C < 2.6 mmol/L，可考虑他汀类药物',
                # 眼科治疗
                'anti_vegf_treatment': {},
                'laser_treatment': {},
                'surgical_treatment': '',
                # 药物
                'medications': [
                    {
                        'name': '羟苯磺酸钙',
                        'category': '眼科用药',
                        'dosage': '500mg 口服',
                        'duration': '3个月',
                        'notes': '改善微循环，可长期服用'
                    },
                    {
                        'name': '复方丹参滴丸',
                        'category': '眼科用药',
                        'dosage': '10粒 口服 tid',
                        'duration': '3个月',
                        'notes': '辅助改善视网膜微循环'
                    }
                ],
                # 生活方式
                'diet_guidance': '''• 严格控制总热量，维持BMI<24
• 减少精制碳水化合物摄入
• 增加深海鱼类摄入（富含Omega-3）
• 补充叶黄素丰富的食物（菠菜、蛋黄）
• 限制高盐饮食（<6g/天）''',
                'exercise_guidance': '''• 每周至少150分钟中等强度有氧运动
• 运动强度以"能说话但不能唱歌"为宜
• 餐后30分钟-1小时进行
• 避免举重、俯卧撑等憋气运动
• 如有血糖过低风险，运动前适当进食''',
                'lifestyle_advice': '''• 严格血糖控制，延缓DR进展
• 定期监测血糖，记录数据
• 每年至少2次眼科检查
• 控制体重和血压
• 避免熬夜和过度用眼''',
                # 随访监测
                'follow_up_plan': {
                    'interval_days': 180,
                    'check_items': ['视力检查', '眼底照相', 'OCT检查', '血糖检测'],
                    'notes': '6-12个月复查一次'
                },
                'monitoring_plan': [
                    {'item': '空腹血糖', 'frequency': '每天', 'target': '4.4-7.0 mmol/L'},
                    {'item': '餐后血糖', 'frequency': '每天', 'target': '<10.0 mmol/L'},
                    {'item': 'HbA1c', 'frequency': '每3个月', 'target': '<7.0%'},
                    {'item': '血压', 'frequency': '每天', 'target': '<130/80 mmHg'},
                    {'item': '眼科检查', 'frequency': '每6-12个月', 'target': '评估DR进展'},
                ],
                'warning_symptoms': '''如出现以下症状，请立即就医：
• 视力明显下降
• 视物变形或变小
• 眼前黑影增多
• 视野缺损''',
                'precautions': '密切随访，避免DR进展'
            }
        )[0]

    def _create_dr2_template(self):
        """中度NPDR"""
        return TreatmentPlanTemplate.objects.update_or_create(
            dr_grade=2,
            defaults={
                'dr_grade': 2,
                'priority': 80,
                'is_active': True,
                'title': '中度NPDR - 积极干预方案',
                # 基础管理目标
                'blood_sugar_target': {
                    'fasting': '4.4-7.0 mmol/L',
                    'postprandial': '<10.0 mmol/L',
                    'hba1c': '<7.0%'
                },
                'blood_pressure_target': {
                    'systolic': '<130 mmHg',
                    'diastolic': '<80 mmHg'
                },
                'lipid_management': 'LDL-C < 2.6 mmol/L，推荐使用他汀类药物',
                # 眼科治疗
                'anti_vegf_treatment': {},
                'laser_treatment': {
                    'type': '格栅样光凝',
                    'indication': '合并黄斑水肿',
                    'sessions': '1-3次',
                    'interval': '间隔2-4周',
                    'notes': '仅针对黄斑水肿区域'
                },
                'surgical_treatment': '',
                # 药物
                'medications': [
                    {
                        'name': '羟苯磺酸钙',
                        'category': '眼科用药',
                        'dosage': '500mg 口服',
                        'duration': '6个月',
                        'notes': '改善微循环，可长期服用'
                    },
                    {
                        'name': '复方丹参滴丸',
                        'category': '眼科用药',
                        'dosage': '10粒 口服 tid',
                        'duration': '3个月',
                        'notes': '辅助改善视网膜微循环'
                    },
                    {
                        'name': '甲钴胺',
                        'category': '营养神经',
                        'dosage': '500μg 口服 tid',
                        'duration': '1个月',
                        'notes': '营养视神经'
                    }
                ],
                # 生活方式
                'diet_guidance': '''• 严格糖尿病饮食
• 控制钠盐摄入（<5g/天）
• 增加膳食纤维（25-30g/天）
• 补充Omega-3脂肪酸
• 避免辛辣刺激性食物''',
                'exercise_guidance': '''• 每周3-5次，每次30分钟
• 中等强度有氧运动
• 避免剧烈运动和头部震动
• 运动时保护眼睛
• 如有不适立即停止''',
                'lifestyle_advice': '''• 严格血糖、血压、血脂控制
• 3-6个月眼科复查一次
• 如有必要，考虑进行激光治疗
• 避免剧烈运动和头部碰撞
• 保持大便通畅，避免用力''',
                # 随访监测
                'follow_up_plan': {
                    'interval_days': 90,
                    'check_items': ['视力检查', '眼底照相', 'OCT检查', 'FFA检查', '血糖血压'],
                    'notes': '3-6个月复查，必要时FFA检查'
                },
                'monitoring_plan': [
                    {'item': '血糖谱', 'frequency': '每天多次', 'target': '控制达标'},
                    {'item': 'HbA1c', 'frequency': '每3个月', 'target': '<7.0%'},
                    {'item': '血压', 'frequency': '每天测量', 'target': '<130/80 mmHg'},
                    {'item': '眼科复查', 'frequency': '每3-6个月', 'target': '监测DR变化'},
                    {'item': '肾功能', 'frequency': '每6个月', 'target': '尿蛋白阴性'},
                ],
                'warning_symptoms': '''如出现以下症状，请立即眼科就诊：
• 视力突然下降
• 眼前大量黑影漂浮
• 视物遮挡感
• 眼睛剧烈疼痛''',
                'precautions': '警惕病情进展至重度NPDR或PDR'
            }
        )[0]

    def _create_dr3_template(self):
        """重度NPDR"""
        return TreatmentPlanTemplate.objects.update_or_create(
            dr_grade=3,
            defaults={
                'dr_grade': 3,
                'priority': 70,
                'is_active': True,
                'title': '重度NPDR - 及时干预方案',
                # 基础管理目标
                'blood_sugar_target': {
                    'fasting': '4.4-7.0 mmol/L',
                    'postprandial': '<10.0 mmol/L',
                    'hba1c': '<7.0%'
                },
                'blood_pressure_target': {
                    'systolic': '<130 mmHg',
                    'diastolic': '<80 mmHg'
                },
                'lipid_management': 'LDL-C < 2.6 mmol/L，规范他汀治疗',
                # 眼科治疗
                'anti_vegf_treatment': {
                    'drug': '雷珠单抗/阿柏西普',
                    'indication': '合并DME或高危特征',
                    'frequency': '每月1次',
                    'course': '3-5次初始化治疗',
                    'notes': '根据病情决定后续治疗'
                },
                'laser_treatment': {
                    'type': '全视网膜激光光凝(PRP)',
                    'indication': '无DME的高危PDR',
                    'sessions': '3-4次',
                    'interval': '间隔1-2周',
                    'notes': '分次完成，避免一次性激光量过大'
                },
                'surgical_treatment': '''指征：
• 玻璃体出血不吸收（3-6个月）
• 牵拉性视网膜脱离累及黄斑
• 视网膜前膜''',
                # 药物
                'medications': [
                    {
                        'name': '羟苯磺酸钙',
                        'category': '眼科用药',
                        'dosage': '500mg 口服',
                        'duration': '6个月',
                        'notes': '改善微循环'
                    },
                    {
                        'name': '复方丹参滴丸',
                        'category': '眼科用药',
                        'dosage': '10粒 口服 tid',
                        'duration': '3个月',
                        'notes': '辅助治疗'
                    }
                ],
                # 生活方式
                'diet_guidance': '''• 严格糖尿病饮食管理
• 低盐低脂饮食
• 适量补充蛋白质
• 避免吸烟饮酒
• 保持大便通畅''',
                'exercise_guidance': '''• 避免剧烈运动
• 避免头部震动性运动
• 可进行轻度散步等
• 避免憋气用力动作
• 注意保护眼睛''',
                'lifestyle_advice': '''• 立即眼科就诊评估治疗指征
• PRP激光或抗VEGF治疗
• 严格控制全身危险因素
• 密切随访，警惕玻璃体出血
• 避免重体力劳动''',
                # 随访监测
                'follow_up_plan': {
                    'interval_days': 60,
                    'check_items': ['视力检查', '眼底检查', 'OCT', '眼压', '全身指标'],
                    'notes': '1-2个月复查一次'
                },
                'monitoring_plan': [
                    {'item': '血糖', 'frequency': '密切监测', 'target': '严格达标'},
                    {'item': '血压', 'frequency': '每天测量', 'target': '<130/80 mmHg'},
                    {'item': '眼科复查', 'frequency': '每1-2个月', 'target': '监测病情'},
                    {'item': '全身检查', 'frequency': '每3个月', 'target': '全面评估'},
                ],
                'warning_symptoms': '''紧急情况！立即就医：
• 突然视力丧失
• 大量黑影漂浮
• 红色幕布样遮挡
• 眼睛剧烈疼痛''',
                'precautions': '重度NPDR随时可能进展为PDR，需密切随访'
            }
        )[0]

    def _create_dr4_template(self):
        """PDR（增殖性DR）"""
        return TreatmentPlanTemplate.objects.update_or_create(
            dr_grade=4,
            defaults={
                'dr_grade': 4,
                'priority': 60,
                'is_active': True,
                'title': 'PDR - 积极治疗方案',
                # 基础管理目标
                'blood_sugar_target': {
                    'fasting': '4.4-7.0 mmol/L',
                    'postprandial': '<10.0 mmol/L',
                    'hba1c': '<7.0-7.5%'
                },
                'blood_pressure_target': {
                    'systolic': '<130 mmHg',
                    'diastolic': '<80 mmHg'
                },
                'lipid_management': 'LDL-C < 1.8 mmol/L，强化他汀治疗',
                # 眼科治疗
                'anti_vegf_treatment': {
                    'drug': '雷珠单抗/阿柏西普/康柏西普',
                    'indication': 'PDR合并/不合并DME',
                    'frequency': '每月1次',
                    'course': '至少3次，后续按需',
                    'notes': '可作为PRP前新辅助治疗'
                },
                'laser_treatment': {
                    'type': '全视网膜激光光凝(PRP)',
                    'indication': '高危PDR',
                    'sessions': '4次或更多',
                    'interval': '间隔1-2周',
                    'notes': '及时完成全量PRP'
                },
                'surgical_treatment': '''玻璃体切割术(PPV)指征：
• 玻璃体出血（3个月不吸收）
• 牵拉性视网膜脱离累及黄斑
• 混合性视网膜脱离
• 玻璃体视网膜界面增殖膜
• 新生血管性青光眼\n治疗时机：越早越好''',
                # 药物
                'medications': [
                    {
                        'name': '羟苯磺酸钙',
                        'category': '眼科用药',
                        'dosage': '500mg 口服',
                        'duration': '长期',
                        'notes': '改善微循环'
                    }
                ],
                # 生活方式
                'diet_guidance': '''• 严格糖尿病饮食
• 避免辛辣刺激食物
• 戒烟戒酒
• 保持心情平稳
• 避免用力排便''',
                'exercise_guidance': '''• 严格避免剧烈运动
• 避免头部低于心脏的运动
• 避免憋气用力
• 禁止举重、俯卧撑
• 可进行轻度活动''',
                'lifestyle_advice': '''• 立即眼科就诊，积极治疗
• PRP激光是基础治疗
• 抗VEGF可辅助治疗
• 必要时及时手术
• 严格控制全身危险因素
• 密切随访，防止复发''',
                # 随访监测
                'follow_up_plan': {
                    'interval_days': 30,
                    'check_items': ['视力', '眼压', '眼底检查', 'OCT', '全身检查'],
                    'notes': '治疗后1个月复查，之后每1-3个月'
                },
                'monitoring_plan': [
                    {'item': '视力', 'frequency': '每天自查', 'target': '监测变化'},
                    {'item': '眼压', 'frequency': '每次复查', 'target': '<21 mmHg'},
                    {'item': '眼科复查', 'frequency': '每1-3个月', 'target': '评估疗效'},
                    {'item': '全身指标', 'frequency': '每月', 'target': '全面达标'},
                ],
                'warning_symptoms': '''紧急情况！立即眼科急诊：
• 突发视力丧失
• 眼前大量黑影
• 红色幕布遮挡
• 眼睛胀痛伴头痛''',
                'precautions': 'PDR是严重威胁视力的并发症，必须积极治疗，密切随访'
            }
        )[0]

