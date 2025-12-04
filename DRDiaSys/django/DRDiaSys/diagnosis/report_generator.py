# -*- coding: utf-8 -*-
"""
PDF诊断报告生成模块
"""
import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from django.conf import settings


class DiagnosisReportGenerator:
    """诊断报告生成器"""
    
    def __init__(self, report, output_path=None):
        self.report = report
        self.diagnosis_task = report.diagnosis_task
        self.patient_image = report.patient_image
        self.patient = self.patient_image.owner
        
        # 设置输出路径
        if output_path is None:
            reports_dir = os.path.join(settings.BASE_DIR, 'media', 'reports')
            os.makedirs(reports_dir, exist_ok=True)
            output_path = os.path.join(reports_dir, f"{report.report_number}.pdf")
        
        self.output_path = output_path
        self.doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=1.2*cm,
            leftMargin=1.2*cm,
            topMargin=1.5*cm,
            bottomMargin=1.2*cm
        )
        
        # 注册中文字体
        self.chinese_font = self._register_chinese_font()
        
        self.styles = self._create_styles()
    def _register_chinese_font(self):
        """注册中文字体，优先使用项目内字体，失败则使用内置STSong"""
        try:
            if hasattr(settings, 'BASE_DIR'):
                base_dir = str(settings.BASE_DIR)
            else:
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

            font_path = os.path.join(base_dir, 'static', 'fonts', 'simsun.ttc')
            if os.path.exists(font_path):
                pdfmetrics.registerFont(TTFont('SimSun', font_path))
                return 'SimSun'
        except Exception as exc:
            print(f"注册 SimSun 字体失败: {exc}")

        try:
            pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
            return 'STSong-Light'
        except Exception as exc:
            print(f"注册 STSong-Light 字体失败: {exc}")

        return 'Helvetica'
    
    def _create_styles(self):
        """创建样式"""
        styles = getSampleStyleSheet()
        
        # 标题样式
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#1a237e'),
            spaceAfter=18,
            alignment=TA_CENTER,
            fontName=self.chinese_font
        )
        
        # 副标题样式
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontSize=12,
            textColor=colors.HexColor('#283593'),
            spaceAfter=8,
            spaceBefore=8,
            fontName=self.chinese_font
        )
        
        # 正文样式
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=10,
            leading=14,
            fontName=self.chinese_font
        )

        # 已复核印章样式
        stamp_style = ParagraphStyle(
            'Stamp',
            parent=styles['Normal'],
            fontSize=14,
            textColor=colors.HexColor('#d32f2f'),
            alignment=TA_RIGHT,
            fontName=self.chinese_font
        )
        
        # 表格标题样式
        table_header_style = ParagraphStyle(
            'TableHeader',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.white,
            alignment=TA_CENTER,
            fontName=self.chinese_font
        )
        
        return {
            'title': title_style,
            'subtitle': subtitle_style,
            'normal': normal_style,
            'table_header': table_header_style,
            'stamp': stamp_style
        }
    
    def _get_patient_info(self):
        """获取患者信息"""
        patient_info = {
            'name': self.patient.username,
            'id': self.patient.id,
        }
        
        # 尝试获取患者详细信息
        try:
            profile = getattr(self.patient, 'patient_info', None)
            if profile:
                patient_info['real_name'] = profile.real_name or ''
                patient_info['gender'] = profile.get_gender_display() if profile.gender else ''
                patient_info['age'] = profile.age or ''
        except:
            pass
        
        return patient_info
    
    def _create_header(self):
        """创建报告头部"""
        elements = []
        
        # 标题
        title = Paragraph("眼底图像AI诊断报告", self.styles['title'])
        elements.append(title)
        elements.append(Spacer(1, 0.3*cm))
        
        patient_info = self._get_patient_info()
        summary_table = [
            ['报告编号', self.report.report_number, '生成时间', datetime.now().strftime('%Y-%m-%d %H:%M')],
            ['患者', patient_info.get('real_name') or patient_info.get('name', ''), '眼别', self.patient_image.get_eye_side_display()],
            ['上传时间', self.patient_image.created_at.strftime('%Y-%m-%d %H:%M'), '报告状态', self.report.get_status_display() if hasattr(self.report, 'get_status_display') else self.report.status],
        ]
        
        report_info_table = Table(summary_table, colWidths=[2.8*cm, 5.2*cm, 2.8*cm, 5.2*cm])
        report_info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f5f7fa')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), self.chinese_font),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.3, colors.HexColor('#dfe3eb')),
        ]))
        
        elements.append(report_info_table)
        return elements
    
    def _create_diagnosis_result_section(self):
        """创建诊断结果部分"""
        elements = []
        
        subtitle = Paragraph("AI诊断摘要", self.styles['subtitle'])
        elements.append(subtitle)
        
        # 诊断摘要
        if self.report.ai_summary:
            summary_para = Paragraph(f"<b>诊断摘要：</b>{self.report.ai_summary}", self.styles['normal'])
            elements.append(summary_para)
            elements.append(Spacer(1, 0.3*cm))
        
        # 病灶统计表格 + 描述
        lesion_stats = self.diagnosis_task.lesion_statistics or {}
        if lesion_stats:
            table_data = [['病灶类型', '占比(%)', '像素数量']]
            detail_rows = []
            for class_id, stat in lesion_stats.items():
                if class_id == 0:
                    continue
                detail_rows.append({
                    'name': stat.get('name', ''),
                    'percentage': stat.get('percentage', 0),
                    'pixels': stat.get('pixel_count', 0)
                })
            detail_rows.sort(key=lambda x: x['percentage'], reverse=True)
            detail_rows = detail_rows[:3] or detail_rows
            for row in detail_rows:
                table_data.append([
                    row['name'],
                    f"{row['percentage']:.2f}",
                    f"{row['pixels']:,}"
                ])
            
            lesion_table = Table(table_data, colWidths=[6*cm, 3*cm, 4*cm])
            lesion_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976d2')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), self.chinese_font),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('TOPPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), self.chinese_font),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
                ('TOPPADDING', (0, 1), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.4, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
            ]))
            
            elements.append(lesion_table)
            elements.append(Spacer(1, 0.2*cm))

            # 文本化描述
            desc_parts = []
            for row in detail_rows:
                desc_parts.append(
                    f"{row['name']}约占视网膜区域的{row['percentage']:.2f}%"
                )
            desc_text = "AI 分割结果提示，本次图像中主要病灶包括：" + "；".join(desc_parts) + "。"
            elements.append(Paragraph(desc_text, self.styles['normal']))
            elements.append(Spacer(1, 0.2*cm))
        else:
            # 没有病灶或仅背景
            no_lesion_text = "AI 分割结果提示：本次图像中未检测到明显病灶。"
            elements.append(Paragraph(no_lesion_text, self.styles['normal']))
            elements.append(Spacer(1, 0.2*cm))
        
        # 结果图像
        if self.diagnosis_task.result_image_path and os.path.exists(self.diagnosis_task.result_image_path):
            try:
                img = Image(self.diagnosis_task.result_image_path, width=14*cm, height=4.2*cm)
                img_caption = Paragraph("<b>图像参考（原图 | 分割 | 叠加）</b>", self.styles['normal'])
                elements.append(img_caption)
                elements.append(Spacer(1, 0.15*cm))
                elements.append(img)
                elements.append(Spacer(1, 0.3*cm))
            except Exception as e:
                print(f"无法添加结果图像: {e}")
        
        return elements
    
    def _create_doctor_review_section(self):
        """创建医生复核部分"""
        elements = []
        
        subtitle = Paragraph("医生复核", self.styles['subtitle'])
        elements.append(subtitle)

        # 已复核印章（仅在报告状态为已确认时展示）
        if self.report.status == 'finalized':
            stamp = Paragraph("【已复核】", self.styles['stamp'])
            elements.append(stamp)
        
        if self.report.reviewed_by:
            review_data = [
                ['复核医生', self.report.reviewed_by.username],
                ['复核时间', self.report.reviewed_at.strftime('%Y-%m-%d %H:%M:%S') if self.report.reviewed_at else ''],
            ]
            
            review_table = Table(review_data, colWidths=[4*cm, 8*cm])
            review_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f5f5f5')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
                ('ALIGN', (1, 0), (1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), self.chinese_font),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            
            elements.append(review_table)

            # 医生具体诊断
            if self.report.doctor_conclusion:
                conclusion_para = Paragraph(f"<b>医生诊断意见：</b>{self.report.doctor_conclusion}", self.styles['normal'])
                elements.append(Spacer(1, 0.2*cm))
                elements.append(conclusion_para)

            if self.report.doctor_notes:
                notes_para = Paragraph(f"<b>医生备注：</b>{self.report.doctor_notes}", self.styles['normal'])
                elements.append(Spacer(1, 0.2*cm))
                elements.append(notes_para)
        else:
            pending_para = Paragraph("待医生复核", self.styles['normal'])
            elements.append(pending_para)
        
        elements.append(Spacer(1, 0.3*cm))
        
        return elements
    
    def _create_footer(self):
        """创建页脚"""
        elements = []
        
        footer_text = Paragraph(
            "<i>本报告由AI系统自动生成，仅供参考。最终诊断结果以医生复核为准。</i>",
            self.styles['normal']
        )
        elements.append(footer_text)
        
        return elements
    
    def generate(self):
        """生成PDF报告"""
        elements = []
        
        # 添加各个部分
        elements.extend(self._create_header())
        elements.extend(self._create_diagnosis_result_section())
        elements.extend(self._create_doctor_review_section())
        elements.extend(self._create_footer())
        
        # 构建PDF
        self.doc.build(elements)
        
        return self.output_path

