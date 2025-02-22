import { pool } from '../config/database.config';

async function testConnection() {
    try {
        console.log('Testing database connection...');
        const client = await pool.connect();
        const result = await client.query('SELECT version()');
        console.log('PostgreSQL version:', result.rows[0].version);
        await client.release();
    } catch (err) {
        console.error('Connection error:', err);
    } finally {
        await pool.end();
    }
}

testConnection();