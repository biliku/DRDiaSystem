<template>
  <div class="dashboard-cover">
    <div class="cover-bg">
      <div class="cover-glow"></div>
      <div class="cover-circuits"></div>
    </div>
    <div class="cover-content">
      <h1 class="cover-title">DR智能诊断系统</h1>
      <p class="cover-desc">
        基于深度学习的糖尿病视网膜病变（DR）智能辅助诊断平台，集成图像管理、自动预处理、病变检测、模型训练与诊断报告一体化，助力医疗智能化升级。
      </p>
      <ul class="cover-features">
        <li>· 一站式视网膜图像管理与处理</li>
        <li>· 高精度病变自动检测与分级</li>
        <li>· 灵活的模型训练与评估</li>
        <li>· 智能辅助诊断与报告生成</li>
        <li>· 多角色权限与数据安全保障</li>
      </ul>
    </div>
  </div>
</template>

<script>
export default {
  name: 'DashboardHome'
}
</script>

<style scoped>
.dashboard-cover {
  width: 100%;
  height: 100%;
  min-height: 480px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #0f2027;
  background: linear-gradient(135deg, #0f2027 0%, #2c5364 100%);
  overflow: hidden;
  position: relative;
}
.cover-bg {
  position: absolute;
  inset: 0;
  z-index: 1;
  pointer-events: none;
}
.cover-glow {
  position: absolute;
  top: 20%;
  left: 50%;
  width: 60vw;
  max-width: 800px;
  height: 30vw;
  max-height: 400px;
  background: radial-gradient(circle, #00c3ff55 0%, #2c536400 80%);
  transform: translate(-50%, -50%);
  filter: blur(40px);
  opacity: 0.7;
}
.cover-circuits {
  position: absolute;
  inset: 0;
  background: repeating-linear-gradient(120deg, #00c3ff22 0 2px, transparent 2px 40px), repeating-linear-gradient(60deg, #00c3ff11 0 1px, transparent 1px 30px);
  opacity: 0.4;
}
.cover-content {
  position: relative;
  z-index: 2;
  text-align: center;
  color: #fff;
  padding: 48px 32px 32px 32px;
  border-radius: 18px;
  background: rgba(20, 40, 60, 0.75);
  box-shadow: 0 8px 32px 0 #00c3ff33;
  width: 100%;
  max-width: 600px;
  min-width: 320px;
  margin: 0 24px;
  backdrop-filter: blur(2px);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}
.cover-title {
  font-size: 2.8rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  margin-bottom: 18px;
  text-shadow: 0 2px 16px #00c3ff88;
}
.cover-desc {
  font-size: 1.18rem;
  margin-bottom: 28px;
  color: #b3e5fc;
}
.cover-features {
  list-style: none;
  padding: 0;
  margin: 0;
  font-size: 1.08rem;
  color: #e1f5fe;
  text-align: left;
  width: 100%;
  max-width: 400px;
  margin-left: auto;
  margin-right: auto;
}
.cover-features li {
  margin-bottom: 10px;
  padding-left: 16px;
  position: relative;
}
.cover-features li:before {
  content: '';
  position: absolute;
  left: 0;
  top: 10px;
  width: 4px;
  height: 4px;
  background: #00c3ff;
  border-radius: 50%;
}
@media (max-width: 600px) {
  .cover-content {
    padding: 24px 8px 16px 8px;
    min-width: unset;
    max-width: 98vw;
  }
  .cover-title {
    font-size: 2rem;
  }
  .cover-features {
    font-size: 0.98rem;
    max-width: 98vw;
  }
}
</style> 