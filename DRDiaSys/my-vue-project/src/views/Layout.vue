<template>
  <!-- 注意：為了確保 el-container 佔滿高度，
              需要全局重置 html, body, #app 的 margin, padding, height -->
  <el-container class="layout-container">
    <!-- 侧边栏 -->
    <el-aside :width="isCollapse ? '64px' : '180px'">
      <div class="logo">
        <!-- Logo 在這裡 -->
        <img src="../assets/logo.svg" alt="Logo" class="logo-img"> <!-- 建議給 img 加個 class 以便樣式區分 -->
        <span v-show="!isCollapse">DR诊断系统</span>
      </div>
      <el-menu
        :default-active="activeMenu"
        class="el-menu-vertical"
        background-color="#304156"
        text-color="#bfcbd9"
        active-text-color="#409EFF"
        :collapse="isCollapse"
        router
      >
        <el-menu-item
          v-for="item in menuItems"
          :key="item.path"
          :index="item.path"
        >
          <el-icon><component :is="item.icon" /></el-icon>
          <span>{{ item.label }}</span>
        </el-menu-item>
      </el-menu>
    </el-aside>

    <!-- 主要内容区 -->
    <el-container class="main-content-container"> <!-- 可以給內層 container 加個 class -->
      <!-- 顶部导航栏 -->
      <el-header>
        <div class="header-left">
          <el-icon class="toggle-sidebar" @click="toggleSidebar">
            <Fold v-if="!isCollapse" />
            <Expand v-else />
          </el-icon>
          <el-breadcrumb separator="/">
            <el-breadcrumb-item :to="{ path: '/' }">首页</el-breadcrumb-item>
            <!-- 確保 currentRoute 有值，避免顯示空麵包屑 -->
            <el-breadcrumb-item v-if="currentRoute">{{ currentRoute }}</el-breadcrumb-item>
          </el-breadcrumb>
        </div>
        <div class="header-right">
          <el-dropdown>
            <span class="user-info">
              <el-avatar :size="32" :src="userAvatar" />
              <span>{{ userName }}</span>
            </span>
            <template #dropdown>
              <el-dropdown-menu>
                <el-dropdown-item>个人信息</el-dropdown-item>
                <el-dropdown-item>修改密码</el-dropdown-item>
                <el-dropdown-item divided @click="logout">退出登录</el-dropdown-item>
              </el-dropdown-menu>
            </template>
          </el-dropdown>
        </div>
      </el-header>
      <!-- 内容区 -->
      <el-main>
        <!-- router-view 加載的內容將會填充這個區域 -->
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<script>
import {
  DataLine,
  Picture,
  Search,
  Connection,
  Document,
  Files,
  User,
  Fold,
  Expand
} from '@element-plus/icons-vue'

export default {
  name: 'MainLayout',
  components: {
    DataLine,
    Picture,
    Search,
    Connection,
    Document,
    Files,
    User,
    Fold,
    Expand
  },
  data() {
    return {
      isCollapse: false,
      userRole: localStorage.getItem('userRole') || 'admin',
      userName: localStorage.getItem('userName') || '管理员',
      userAvatar: 'https://cube.elemecdn.com/0/88/03b0d39583f48206768a7534e55bcpng.png '
    }
  },
  computed: {
    activeMenu() {
      // 如果當前路由是根路徑，指向 dashboard 作為默認激活菜單
      if (this.$route.path === '/') {
          // 你可能需要將 '/' 重定向到 '/dashboard' 在你的路由配置中
          // 或者在這裡手動指定 default-active 為 '/dashboard' 如果 '/' 是登錄頁或其他非主頁面
           return '/dashboard'; // 假設 dashboard 是首頁
      }
      return this.$route.path
    },
    currentRoute() {
      const routeMap = {
        '/dashboard': '首页',
        '/image-processing': '图像管理与预处理',
        '/disease-detection': '病变检测',
        '/model-training': '模型训练',
        '/auxiliary-diagnosis': '辅助诊断',
        '/report-generation': '诊断报告',
        '/data-management': '数据管理',
        '/user-management': '用户管理'
      }
      // 根據當前路由找到對應的名稱，如果不存在則返回空字符串
      return routeMap[this.$route.path] || '';
    },
    // 根据角色过滤菜单项
    menuItems() {
      const allMenus = [
        { path: '/dashboard', icon: 'DataLine', label: '首页' },
        { path: '/image-processing', icon: 'Picture', label: '图像管理与预处理' },
        { path: '/model-training', icon: 'Connection', label: '模型训练' },
        { path: '/disease-detection', icon: 'Search', label: '病变检测' },
        { path: '/auxiliary-diagnosis', icon: 'User', label: '辅助诊断' },
        { path: '/report-generation', icon: 'Document', label: '诊断报告' },
        { path: '/data-management', icon: 'Files', label: '数据管理' },
        { path: '/user-management', icon: 'User', label: '用户管理' }
      ]

      // 保留原來的角色過濾邏輯，如果需要
      const roleMenus = {
        admin: ['/dashboard', '/image-processing', '/model-training', '/disease-detection', '/report-generation', '/data-management', '/user-management'],
        doctor: ['/dashboard', '/auxiliary-diagnosis', '/report-generation'],
        patient: ['/dashboard', '/report-generation']
      }
      
      if (this.userRole && roleMenus[this.userRole]) {
          return allMenus.filter(menu => roleMenus[this.userRole].includes(menu.path));
      }
      
      // 如果沒有角色或角色無效，返回所有菜單
      return allMenus;
    }
  },
  methods: {
    toggleSidebar() {
      this.isCollapse = !this.isCollapse
      // 使用 nextTick 確保 DOM 更新後再觸發 resize
      // 這對於某些基於尺寸計算的組件 (如一些圖表庫) 可能有用，但對於基本布局通常不是必需的
      // this.$nextTick(() => {
      //   window.dispatchEvent(new Event('resize'))
      // })
    },

    // 退出登录
    logout() {
      localStorage.removeItem('token')
      localStorage.removeItem('userRole')
      localStorage.removeItem('userName')
      // 替換當前路由，避免用戶回退到需要登錄的頁面
      this.$router.replace('http://localhost:8080/login/')
      // 顯示消息提示可能需要 Element Plus 的 ElMessage 組件
      // this.$message.success('已退出登录') // 如果你配置了 ElMessage
    }
  },
  created() {
    // 檢查登錄狀態
    const token = localStorage.getItem('token')
    if (!token && this.$route.path !== '/login') {
      // 如果沒有 token 且當前不在登錄頁，則重定向到登錄頁
      this.$router.replace('/login')
    }
    // 確保在組件創建時獲取用戶信息
    this.userName = localStorage.getItem('userName') || '管理员';
    this.userRole = localStorage.getItem('userRole') || 'admin';

    // 如果當前路徑是根路徑 '/' 且不是登錄頁，可能需要重定向到默認的首頁，例如 '/dashboard'
    if (this.$route.path === '/' && this.$route.path !== '/login') {
        // this.$router.replace('/dashboard'); // 確保 '/' 指向一個有效的首頁
    }
  },
  // 可以在 watch 中監聽路由變化來更新麵包屑，但 computed 已經實現了
  // watch: {
  //     '$route'(to, from) {
  //         // 可選：在這裡做一些路由切換時的操作
  //     }
  // }
}
</script>

<style scoped>
/* 確保根容器佔滿父元素（通常是 #app） */
.layout-container {
  height: 100vh; /* 或者 100% 如果 html, body, #app 設置了 height: 100% */
  display: flex; /* el-container 默認就是 flex */
  flex-direction: row; /* el-container 默認就是 row */
}

.el-aside {
  background-color: #304156;
  transition: width 0.3s;
  overflow-x: hidden; /* 防止側邊欄內容溢出導致橫向滾動條 */
  flex-shrink: 0; /* 防止側邊欄被壓縮 */
}

.logo {
  height: 60px;
  display: flex;
  align-items: center;
  padding: 0 20px;
  color: #fff;
  font-size: 18px;
  font-weight: bold;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis; /* logo 文字溢出時顯示省略號 */
}

.logo-img { /* 給 logo img 添加樣式 */
  width: 32px;
  height: 32px;
  margin-right: 12px;
  flex-shrink: 0; /* 防止圖片縮小 */
}



.el-menu-vertical {
    border-right: none; /* 移除菜單默認的右邊框 */
    height: calc(100vh - 60px); /* 讓菜單高度填充 aside 除 logo 外的剩餘空間 */
    overflow-y: auto; /* 菜單項過多時，在菜單內部滾動 */
}


/* 主要內容區容器，包含 header 和 main */
.main-content-container {
    flex-direction: column; /* el-container 內嵌時默認是 column */
    flex-grow: 1; /* 讓它佔滿 aside 之外的剩餘寬度 */
    overflow: hidden; /* 防止內部元素溢出影響布局 */
}


.el-header {
  background-color: #fff;
  border-bottom: 1px solid #e6e6e6;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  height: 60px; /* 確保 header 高度固定 */
  flex-shrink: 0; /* 防止 header 被壓縮 */
}

.header-left {
  display: flex;
  align-items: center;
  gap: 20px; /* 使用 gap 來代替硬編碼的 margin */
}

.toggle-sidebar {
  font-size: 20px;
  cursor: pointer;
  display: flex; /* 確保圖標居中 */
  align-items: center;
  justify-content: center;
}

.header-right {
  display: flex;
  align-items: center;
}

.user-info {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  padding: 0 10px; /* 增加點擊區域 */
}

/* 移除 el-main 的默認 padding */
.el-main {
  background-color: #f0f2f5;
  padding: 0; /* <--- 關鍵修改：移除默認 padding */
  flex-grow: 1; /* 確保 main 佔據 header 下方所有剩餘空間 */
  overflow: auto; /* <--- 關鍵修改：當內容超出時，在 main 區域內部出現滾動條 */
}

/* Element Plus 菜單項內部樣式，確保圖標和文字對齊 */
:deep(.el-menu-item) {
  display: flex;
  align-items: center;
  /* gap: 8px; /* Element Plus 默認就有 padding-left，使用 gap 可能會導致雙重間距 */
  white-space: nowrap;
}

/* 調整菜單項內部圖標和文字的間距，避免使用 gap 和 Element Plus 默認樣式衝突 */
:deep(.el-menu-item > .el-icon) {
  margin-right: 10px; /* 調整圖標和文字間距 */
  flex-shrink: 0;
}

:deep(.el-menu-item span) {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* 調整折疊狀態下菜單項的 tooltip 樣式 */
.el-menu--collapse  .el-menu-vertical .el-tooltip__trigger {
    padding: 0 20px; /* 調整折疊菜單項居中 */
}
</style>
